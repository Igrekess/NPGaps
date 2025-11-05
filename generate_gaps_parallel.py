#!/usr/bin/env python3
"""
Générateur de gaps parallèle - Version corrigée
Corrections: gestion connexions + validation gaps + gestion mémoire
"""

import argparse
import json
import os
import sys
import time
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import tempfile
import multiprocessing as mp
from multiprocessing import Pool
from typing import Tuple, List, Dict, Optional
import psutil


def generate_primes_chunk(start: int, stop: int) -> np.ndarray:
    """Génère les nombres premiers dans [start, stop]"""
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt') as tmp:
        tmp_path = tmp.name
    
    try:
        cmd = ['primesieve', str(start), str(stop), '--print']
        with open(tmp_path, 'w') as outfile:
            result = subprocess.run(cmd, stdout=outfile, stderr=subprocess.PIPE, 
                                   text=True, timeout=3600)
        
        if result.returncode != 0:
            raise RuntimeError(f"primesieve error: {result.stderr}")
        
        primes = []
        with open(tmp_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and line.isdigit():
                    primes.append(int(line))
        
        return np.array(primes, dtype=np.uint64)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def encode_gaps(gaps: np.ndarray) -> np.ndarray:
    """Encode les gaps en uint8"""
    result = []
    for gap in gaps:
        gap_int = int(gap)
        if gap_int < 255:
            result.append(gap_int)
        else:
            result.append(255)
            result.append((gap_int >> 8) & 0xFF)
            result.append(gap_int & 0xFF)
    return np.array(result, dtype=np.uint8)


def process_chunk_worker(chunk_info: Tuple[int, int, int]) -> Dict:
    """Worker - génère les premiers d'un chunk (sans calcul de gaps)"""
    chunk_id, start, stop = chunk_info
    
    try:
        primes = generate_primes_chunk(start, stop)
        
        if len(primes) == 0:
            return {
                'chunk_id': chunk_id,
                'success': True,
                'num_primes': 0,
                'primes': np.array([], dtype=np.uint64)
            }
        
        return {
            'chunk_id': chunk_id,
            'success': True,
            'num_primes': len(primes),
            'primes': primes
        }
    except Exception as e:
        return {
            'chunk_id': chunk_id,
            'success': False,
            'error': str(e)
        }


class ParallelGapsGenerator:
    """Générateur parallèle avec gestion correcte des connexions"""
    
    def __init__(self, target, start=None, output_dir="gaps_data", 
                 num_workers=None, buffer_size_gb=32):
        self.target = int(target)
        self.start_number = int(start) if start is not None else 2
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        cpu_count = mp.cpu_count()
        self.num_workers = num_workers if num_workers else max(cpu_count - 2, 1)
        # Limiter les workers au nombre de CPU
        if self.num_workers > cpu_count:
            print(f"⚠️  Réduction workers de {self.num_workers} à {cpu_count} (CPU disponibles)")
            self.num_workers = cpu_count
        
        self.buffer_size_bytes = buffer_size_gb * 1024**3
        self.check_primesieve()
        
        # Taille des chunks
        interval_size = self.target - self.start_number
        target_chunks_per_worker = 150
        total_chunks = self.num_workers * target_chunks_per_worker
        self.chunk_size = max(int(interval_size / total_chunks), int(1e9))
        
        if self.chunk_size >= 1e11:
            self.chunk_size = int(np.round(self.chunk_size / 1e11) * 1e11)
        elif self.chunk_size >= 1e10:
            self.chunk_size = int(np.round(self.chunk_size / 1e10) * 1e10)
        else:
            self.chunk_size = int(np.round(self.chunk_size / 1e9) * 1e9)
        
        # Fichiers
        if self.start_number > 2:
            file_suffix = f"{self.start_number:.0e}_to_{self.target:.0e}"
        else:
            file_suffix = f"to_{self.target:.0e}"
        
        self.gaps_file = self.output_dir / f"gaps_{file_suffix}.dat"
        self.metadata_file = self.output_dir / f"metadata_{file_suffix}.json"
        self.checkpoint_file = self.output_dir / f"checkpoint_{file_suffix}.json"
        
        self.stats = {
            'start_time': None,
            'end_time': None,
            'total_gaps': 0,
            'total_bytes': 0,
            'chunks_processed': 0,
            'total_chunks': 0,
            'first_prime': None,
            'last_prime': None
        }
    
    def check_primesieve(self):
        try:
            result = subprocess.run(['primesieve', '--version'], 
                                   capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print(f"✓ primesieve: {result.stdout.strip()}")
            else:
                raise FileNotFoundError
        except:
            print("❌ primesieve non installé")
            sys.exit(1)
    
    def prepare_chunks(self) -> List[Tuple[int, int, int]]:
        """Prépare les chunks (sans prev_prime)"""
        chunks = []
        position = self.start_number
        chunk_id = 0
        
        while position < self.target:
            chunk_start = position
            chunk_stop = min(position + self.chunk_size, self.target)
            chunks.append((chunk_id, chunk_start, chunk_stop))
            position = chunk_stop
            chunk_id += 1
        
        self.stats['total_chunks'] = len(chunks)
        return chunks
    
    def generate_all_chunks(self, chunks: List[Tuple]) -> List[Dict]:
        """Génère tous les chunks en parallèle"""
        print("\n🔗 Phase 1/2: Génération des premiers...")
        start_time = time.time()
        
        with Pool(processes=self.num_workers) as pool:
            results = []
            completed = 0
            last_update = time.time()
            
            for result in pool.imap_unordered(process_chunk_worker, chunks):
                results.append(result)
                completed += 1
                
                current_time = time.time()
                if completed % 10 == 0 or (current_time - last_update) >= 1.0:
                    elapsed = current_time - start_time
                    progress = (completed / len(chunks)) * 100
                    speed = completed / elapsed if elapsed > 0 else 0
                    
                    if progress > 0:
                        eta = (elapsed / progress) * (100 - progress)
                        eta_str = str(timedelta(seconds=int(eta)))
                    else:
                        eta_str = "..."
                    
                    print(f"\r  ⏳ {completed}/{len(chunks)} ({progress:5.1f}%) | "
                          f"{speed:5.1f} c/s | ETA: {eta_str}",
                          end='', flush=True)
                    last_update = current_time
        
        elapsed = time.time() - start_time
        print(f"\r  ✅ {len(results)}/{len(chunks)} chunks en {str(timedelta(seconds=int(elapsed)))}     ")
        
        # Trier par chunk_id
        results.sort(key=lambda x: x['chunk_id'])
        return results
    
    def compute_gaps_sequential(self, results: List[Dict]) -> List[Dict]:
        """Calcule les gaps séquentiellement avec validation"""
        print("\n🔗 Phase 2/3: Calcul des gaps...")
        start_time = time.time()
        
        # Trouver le dernier premier avant start_number si nécessaire
        prev_prime = None
        if self.start_number > 2:
            search_start = max(2, self.start_number - 1000)
            primes_before = generate_primes_chunk(search_start, self.start_number)
            if len(primes_before) > 0:
                prev_prime = int(primes_before[-1])
        
        gap_results = []
        
        for i, result in enumerate(results):
            if not result['success']:
                print(f"\n❌ Chunk {result['chunk_id']}: {result.get('error')}")
                continue
            
            if result['num_primes'] == 0:
                continue
            
            primes = result['primes']
            
            # Calculer les gaps pour ce chunk
            if prev_prime is not None:
                # Gap de connexion + gaps internes
                all_primes = np.concatenate([[prev_prime], primes])
                gaps = np.diff(all_primes)
            else:
                # Premier chunk, pas de gap de connexion
                gaps = np.diff(primes)
            
            # VALIDATION: tous les gaps doivent être > 0
            if np.any(gaps <= 0):
                neg_gaps = gaps[gaps <= 0]
                print(f"\n❌ ERREUR chunk {result['chunk_id']}: gaps négatifs détectés!")
                print(f"   Gaps négatifs: {neg_gaps[:5]}")
                print(f"   prev_prime: {prev_prime}")
                print(f"   first_prime: {primes[0] if len(primes) > 0 else 'N/A'}")
                continue
            
            # Encoder
            gaps_encoded = encode_gaps(gaps)
            
            gap_results.append({
                'chunk_id': result['chunk_id'],
                'success': True,
                'num_gaps': len(gaps),
                'gaps_encoded': gaps_encoded,
                'first_prime': int(primes[0]),
                'last_prime': int(primes[-1]),
                'bytes_size': len(gaps_encoded)
            })
            
            # Mettre à jour prev_prime pour le prochain chunk
            prev_prime = int(primes[-1])
            
            # Affichage
            if (i + 1) % 100 == 0 or (i + 1) == len(results):
                elapsed = time.time() - start_time
                progress = ((i + 1) / len(results)) * 100
                speed = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"\r  ⏳ {i+1}/{len(results)} ({progress:5.1f}%) | {speed:5.1f} c/s",
                      end='', flush=True)
        
        elapsed = time.time() - start_time
        print(f"\r  ✅ {len(gap_results)} chunks calculés en {str(timedelta(seconds=int(elapsed)))}     ")
        return gap_results
    
    def format_time(self, seconds: float) -> str:
        return str(timedelta(seconds=int(seconds)))
    
    def format_bytes(self, bytes_count: int) -> str:
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_count < 1024:
                return f"{bytes_count:.2f} {unit}"
            bytes_count /= 1024
        return f"{bytes_count:.2f} PB"
    
    def save_checkpoint(self, completed: int):
        checkpoint = {
            'completed_chunks': completed,
            'total_chunks': self.stats['total_chunks'],
            'progress': (completed / self.stats['total_chunks']) * 100,
            'timestamp': datetime.now().isoformat(),
            'stats': self.stats
        }
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    def load_checkpoint(self) -> Optional[int]:
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                print(f"✓ Checkpoint: {checkpoint['completed_chunks']}/{checkpoint['total_chunks']} chunks")
                if 'stats' in checkpoint:
                    self.stats = checkpoint['stats']
                return checkpoint['completed_chunks']
            except Exception as e:
                print(f"⚠ Erreur checkpoint: {e}")
        return None
    
    def generate(self):
        print("=" * 80)
        print("🚀 GÉNÉRATION PARALLÈLE DE GAPS")
        print("=" * 80)
        print(f"Intervalle: {self.start_number:.2e} → {self.target:.2e}")
        print(f"Workers: {self.num_workers}")
        print(f"Chunk: {self.chunk_size:.2e}")
        print(f"Fichier: {self.gaps_file}")
        print("=" * 80)
        
        # Préparer chunks
        print(f"\n📦 Préparation...")
        chunks = self.prepare_chunks()
        print(f"✓ {len(chunks)} chunks")
        
        # Checkpoint
        completed_checkpoint = self.load_checkpoint()
        if completed_checkpoint:
            chunks = chunks[completed_checkpoint:]
            print(f"✓ Reprise depuis chunk {completed_checkpoint}")
        
        self.stats['start_time'] = datetime.now().isoformat()
        start_time = time.time()
        
        try:
            mode = 'ab' if completed_checkpoint else 'wb'
            
            with open(self.gaps_file, mode) as f:
                # Phase 1: Générer les premiers
                prime_results = self.generate_all_chunks(chunks)
                
                # Phase 2: Calculer les gaps séquentiellement
                gap_results = self.compute_gaps_sequential(prime_results)
                
                # Phase 3: Écrire
                print("\n💾 Phase 3/3: Écriture...")
                start_write = time.time()
                
                for i, result in enumerate(gap_results):
                    f.write(result['gaps_encoded'].tobytes())
                    
                    self.stats['total_gaps'] += result['num_gaps']
                    self.stats['total_bytes'] += result['bytes_size']
                    
                    if self.stats['first_prime'] is None:
                        self.stats['first_prime'] = result['first_prime']
                    self.stats['last_prime'] = result['last_prime']
                    
                    if (i + 1) % 100 == 0 or (i + 1) == len(gap_results):
                        elapsed = time.time() - start_write
                        progress = ((i + 1) / len(gap_results)) * 100
                        speed = (i + 1) / elapsed if elapsed > 0 else 0
                        print(f"\r  ⏳ {i+1}/{len(gap_results)} ({progress:5.1f}%) | "
                              f"{self.format_bytes(self.stats['total_bytes'])} | {speed:5.1f} c/s",
                              end='', flush=True)
                    
                    if (i + 1) % 500 == 0:
                        self.save_checkpoint(i + 1 + (completed_checkpoint or 0))
                
                elapsed_write = time.time() - start_write
                print(f"\r  ✅ {len(gap_results)} chunks écrits en {self.format_time(elapsed_write)} | "
                      f"{self.format_bytes(self.stats['total_bytes'])}     ")
            
            self.stats['end_time'] = datetime.now().isoformat()
            
        except KeyboardInterrupt:
            print("\n\n⚠ Interruption")
            self.save_checkpoint(self.stats['chunks_processed'])
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ Erreur: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
        
        self.save_metadata()
        self.display_final_stats(start_time)
    
    def display_final_stats(self, start_time: float):
        elapsed = time.time() - start_time
        
        print("\n" + "=" * 80)
        print("✅ TERMINÉ")
        print("=" * 80)
        print(f"Intervalle: {self.start_number:.2e} → {self.target:.2e}")
        print(f"Gaps: {self.stats['total_gaps']:,}")
        print(f"Premier: {self.stats['first_prime']}")
        print(f"Dernier: {self.stats['last_prime']}")
        print(f"Taille: {self.format_bytes(self.stats['total_bytes'])}")
        print(f"Chunks: {self.stats['total_chunks']}")
        print(f"Workers: {self.num_workers}")
        print(f"Temps: {self.format_time(elapsed)}")
        print(f"Vitesse: {(self.target - self.start_number) / elapsed:.2e} nombres/s")
        print(f"\n📁 {self.gaps_file}")
        print("=" * 80)
    
    def save_metadata(self):
        metadata = {
            'target': self.target,
            'start_number': self.start_number,
            'chunk_size': self.chunk_size,
            'num_workers': self.num_workers,
            'first_prime': self.stats['first_prime'],
            'last_prime': self.stats['last_prime'],
            'total_gaps': self.stats['total_gaps'],
            'total_bytes': self.stats['total_bytes'],
            'chunks_processed': self.stats['total_chunks'],
            'start_time': self.stats['start_time'],
            'end_time': self.stats['end_time'],
            'gaps_file': str(self.gaps_file),
            'encoding': 'uint8 with 255 marker',
            'parallel_mode': True
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Génération parallèle de gaps")
    parser.add_argument('--target', type=float, required=True)
    parser.add_argument('--start', type=float)
    parser.add_argument('--output', type=str, default='gaps_data')
    parser.add_argument('--workers', type=int)
    parser.add_argument('--buffer', type=int, default=32)
    
    args = parser.parse_args()
    
    cpu_count = mp.cpu_count()
    ram_gb = psutil.virtual_memory().total / 1024**3
    
    print(f"\n💻 Ressources:")
    print(f"  CPU: {cpu_count} cores")
    print(f"  RAM: {ram_gb:.1f} GB\n")
    
    generator = ParallelGapsGenerator(
        target=args.target,
        start=args.start,
        output_dir=args.output,
        num_workers=args.workers,
        buffer_size_gb=args.buffer
    )
    
    generator.generate()


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
