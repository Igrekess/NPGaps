#!/usr/bin/env python3
"""
Générateur de gaps parallèle optimisé pour 32+ cores - Version Corrigée

Corrections:
- Résolution du problème de pickling (pas de Manager)
- Sérialisation JSON correcte
- Fonction worker standalone

Auteur: Pour le projet Théorie de la Persistance
Date: 2025-11-05
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


# ============================================================================
# FONCTIONS WORKERS (doivent être au top-level pour pickling)
# ============================================================================

def generate_primes_chunk(start: int, stop: int) -> np.ndarray:
    """
    Génère les nombres premiers dans [start, stop] via primesieve CLI
    Fonction standalone pour multiprocessing
    """
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
    """Encode les gaps en uint8 avec marqueur 255"""
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


def process_chunk_worker(chunk_info: Tuple[int, int, int, Optional[int]]) -> Dict:
    """
    Worker fonction - traite un chunk de nombres
    Fonction standalone pour éviter les problèmes de pickling
    
    Args:
        chunk_info: (chunk_id, start, stop, prev_prime)
        
    Returns:
        dict: Résultats du chunk
    """
    chunk_id, start, stop, prev_prime = chunk_info
    
    try:
        # Générer les premiers du chunk
        primes = generate_primes_chunk(start, stop)
        
        if len(primes) == 0:
            return {
                'chunk_id': chunk_id,
                'success': True,
                'num_primes': 0,
                'gaps_encoded': np.array([], dtype=np.uint8),
                'first_prime': None,
                'last_prime': None
            }
        
        # Calculer les gaps
        if prev_prime is not None:
            gaps = np.diff(np.concatenate([[prev_prime], primes]))
        else:
            gaps = np.diff(primes)
        
        # Encoder
        gaps_encoded = encode_gaps(gaps)
        
        return {
            'chunk_id': chunk_id,
            'success': True,
            'num_primes': len(primes),
            'num_gaps': len(gaps),
            'gaps_encoded': gaps_encoded,
            'first_prime': int(primes[0]),
            'last_prime': int(primes[-1]),
            'bytes_size': len(gaps_encoded)
        }
        
    except Exception as e:
        return {
            'chunk_id': chunk_id,
            'success': False,
            'error': str(e)
        }


# ============================================================================
# CLASSE PRINCIPALE
# ============================================================================

class ParallelGapsGenerator:
    """Générateur de gaps parallèle sans Manager (évite pickling)"""
    
    def __init__(self, target, start=None, output_dir="gaps_data", 
                 num_workers=None, buffer_size_gb=32):
        """
        Args:
            target: Nombre cible (ex: 1e15)
            start: Nombre de départ (ex: 1e12). Si None, démarre à 2
            output_dir: Répertoire de sortie
            num_workers: Nombre de workers (défaut: CPU count - 2)
            buffer_size_gb: Taille du buffer en mémoire (GB)
        """
        self.target = int(target)
        self.start_number = int(start) if start is not None else 2
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration parallélisation
        cpu_count = mp.cpu_count()
        self.num_workers = num_workers if num_workers else max(cpu_count - 2, 1)
        self.buffer_size_bytes = buffer_size_gb * 1024**3
        
        # Vérifier primesieve
        self.check_primesieve()
        
        # Calcul de la taille des chunks
        interval_size = self.target - self.start_number
        target_chunks_per_worker = 150
        total_chunks = self.num_workers * target_chunks_per_worker
        self.chunk_size = max(int(interval_size / total_chunks), int(1e9))
        
        # Arrondir
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
        
        # Statistiques (dict simple, pas de Manager)
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
        """Vérifie que primesieve est installé"""
        try:
            result = subprocess.run(['primesieve', '--version'], 
                                   capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                version = result.stdout.strip()
                print(f"✓ primesieve détecté: {version}")
            else:
                raise FileNotFoundError
        except (FileNotFoundError, subprocess.TimeoutExpired):
            print("❌ ERREUR: primesieve n'est pas installé")
            print("\nInstallation: pip install primesieve")
            sys.exit(1)
    
    def prepare_chunks(self) -> List[Tuple[int, int, int, Optional[int]]]:
        """Prépare la liste des chunks à traiter"""
        chunks = []
        position = self.start_number
        chunk_id = 0
        
        # Dernier premier avant start_number si besoin
        prev_prime = None
        if self.start_number > 2:
            search_start = max(2, self.start_number - 1000)
            primes_before = generate_primes_chunk(search_start, self.start_number)
            if len(primes_before) > 0:
                prev_prime = int(primes_before[-1])
        
        # Créer les chunks
        while position < self.target:
            chunk_start = position
            chunk_stop = min(position + self.chunk_size, self.target)
            
            chunks.append((chunk_id, chunk_start, chunk_stop, prev_prime))
            
            prev_prime = None  # Sera mis à jour dynamiquement
            position = chunk_stop
            chunk_id += 1
        
        self.stats['total_chunks'] = len(chunks)
        
        return chunks
    
    def generate_sequential_chunks(self, chunks: List[Tuple]) -> List[Dict]:
        """
        Génère tous les chunks en parallèle puis les reconstitue
        """
        print("\n🔗 Phase 1/2: Génération des chunks en parallèle...")
        
        # Générer en parallèle avec contexte 'spawn' pour éviter les problèmes
        with Pool(processes=self.num_workers) as pool:
            results = []
            completed = 0
            
            # imap pour avoir les résultats au fur et à mesure
            for result in pool.imap_unordered(process_chunk_worker, chunks):
                results.append(result)
                completed += 1
                
                if completed % 100 == 0 or completed == len(chunks):
                    progress = (completed / len(chunks)) * 100
                    print(f"\r  Chunks générés: {completed}/{len(chunks)} ({progress:.1f}%)", 
                          end='', flush=True)
        
        print(f"\r  Chunks générés: {len(results)}/{len(chunks)} ✓      ")
        
        # Trier par chunk_id
        results.sort(key=lambda x: x['chunk_id'])
        
        print("🔗 Phase 2/2: Reconstruction des connexions...")
        
        # Reconstruire les gaps de connexion
        sequential_results = []
        prev_last_prime = None
        
        for i, result in enumerate(results):
            if not result['success']:
                print(f"\n❌ Erreur chunk {result['chunk_id']}: {result.get('error', 'Unknown')}")
                continue
            
            if result['num_primes'] == 0:
                continue
            
            # Recalculer le premier gap si nécessaire
            if prev_last_prime is not None and result['first_prime'] is not None:
                connection_gap = result['first_prime'] - prev_last_prime
                
                # Décoder
                gaps_decoded = self.decode_gaps(result['gaps_encoded'])
                
                if len(gaps_decoded) > 0:
                    gaps_decoded[0] = connection_gap
                
                # Re-encoder
                result['gaps_encoded'] = encode_gaps(gaps_decoded)
                result['bytes_size'] = len(result['gaps_encoded'])
            
            sequential_results.append(result)
            
            if result['last_prime'] is not None:
                prev_last_prime = result['last_prime']
            
            if (i + 1) % 100 == 0:
                print(f"\r  Connexions: {i+1}/{len(results)}", end='', flush=True)
        
        print(f"\r  Connexions: {len(results)}/{len(results)} ✓      ")
        
        return sequential_results
    
    def decode_gaps(self, encoded: np.ndarray) -> np.ndarray:
        """Décode les gaps"""
        gaps = []
        i = 0
        while i < len(encoded):
            if encoded[i] < 255:
                gaps.append(encoded[i])
                i += 1
            else:
                high = encoded[i + 1]
                low = encoded[i + 2]
                gap = (high << 8) | low
                gaps.append(gap)
                i += 3
        
        return np.array(gaps, dtype=np.uint64)
    
    def save_checkpoint(self, completed_chunks: int):
        """Sauvegarde un checkpoint"""
        checkpoint = {
            'completed_chunks': completed_chunks,
            'total_chunks': self.stats['total_chunks'],
            'progress': (completed_chunks / self.stats['total_chunks']) * 100,
            'timestamp': datetime.now().isoformat(),
            'stats': self.stats  # Déjà un dict simple
        }
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    def load_checkpoint(self) -> Optional[int]:
        """Charge le checkpoint s'il existe"""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                
                print(f"✓ Checkpoint trouvé: {checkpoint['completed_chunks']}/{checkpoint['total_chunks']} chunks")
                print(f"  Progression: {checkpoint['progress']:.1f}%")
                
                # Restaurer les stats
                if 'stats' in checkpoint:
                    self.stats = checkpoint['stats']
                
                return checkpoint['completed_chunks']
            except Exception as e:
                print(f"⚠ Erreur lecture checkpoint: {e}")
                return None
        return None
    
    def format_time(self, seconds: float) -> str:
        """Formate un temps"""
        return str(timedelta(seconds=int(seconds)))
    
    def format_bytes(self, bytes_count: int) -> str:
        """Formate une taille"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB', 'PB']:
            if bytes_count < 1024:
                return f"{bytes_count:.2f} {unit}"
            bytes_count /= 1024
        return f"{bytes_count:.2f} EB"
    
    def display_progress(self, completed_chunks: int, start_time: float):
        """Affiche la progression"""
        progress = (completed_chunks / self.stats['total_chunks']) * 100
        elapsed = time.time() - start_time
        
        if progress > 0:
            eta_seconds = (elapsed / progress) * (100 - progress)
            eta = self.format_time(eta_seconds)
        else:
            eta = "Calcul..."
        
        # RAM usage
        process = psutil.Process()
        ram_gb = process.memory_info().rss / 1024**3
        
        print(f"\r📊 {progress:6.2f}% | "
              f"Chunks: {completed_chunks}/{self.stats['total_chunks']} | "
              f"Gaps: {self.stats['total_gaps']:,} | "
              f"Workers: {self.num_workers} | "
              f"RAM: {ram_gb:.1f} GB | "
              f"Écrit: {self.format_bytes(self.stats['total_bytes'])} | "
              f"ETA: {eta}",
              end='', flush=True)
    
    def generate(self):
        """Génération principale"""
        print("=" * 80)
        print("🚀 GÉNÉRATION PARALLÈLE DE GAPS")
        print("=" * 80)
        print(f"Intervalle: {self.start_number:.2e} → {self.target:.2e}")
        print(f"Workers: {self.num_workers}")
        print(f"Taille chunk: {self.chunk_size:.2e}")
        print(f"Buffer mémoire: {self.buffer_size_bytes / 1024**3:.1f} GB")
        print(f"Fichier sortie: {self.gaps_file}")
        print("=" * 80)
        
        # Préparer les chunks
        print(f"\n📦 Préparation des chunks...")
        chunks = self.prepare_chunks()
        print(f"✓ {len(chunks)} chunks préparés")
        
        # Vérifier checkpoint
        completed_checkpoint = self.load_checkpoint()
        if completed_checkpoint:
            chunks = chunks[completed_checkpoint:]
            print(f"✓ Reprise à partir du chunk {completed_checkpoint}")
        
        self.stats['start_time'] = datetime.now().isoformat()
        start_time = time.time()
        
        try:
            # Mode d'ouverture
            mode = 'ab' if completed_checkpoint else 'wb'
            
            with open(self.gaps_file, mode) as f:
                # Générer les chunks
                sequential_results = self.generate_sequential_chunks(chunks)
                
                # Écrire les résultats
                print("\n💾 Écriture des résultats...")
                batch = []
                batch_size_bytes = 0
                max_batch_bytes = 1024**3  # 1 GB
                
                for i, result in enumerate(sequential_results):
                    if not result['success'] or len(result['gaps_encoded']) == 0:
                        continue
                    
                    batch.append(result)
                    batch_size_bytes += result['bytes_size']
                    
                    # Écrire par batch
                    if batch_size_bytes >= max_batch_bytes or len(batch) >= 100:
                        # Écrire le batch
                        for r in batch:
                            f.write(r['gaps_encoded'].tobytes())
                            
                            self.stats['total_gaps'] += r['num_gaps']
                            self.stats['total_bytes'] += r['bytes_size']
                            
                            if self.stats['first_prime'] is None:
                                self.stats['first_prime'] = r['first_prime']
                            
                            if r['last_prime'] is not None:
                                self.stats['last_prime'] = r['last_prime']
                        
                        batch = []
                        batch_size_bytes = 0
                        
                        # Update progress
                        completed = i + 1 + (completed_checkpoint or 0)
                        self.display_progress(completed, start_time)
                        
                        # Checkpoint
                        if completed % 500 == 0:
                            self.save_checkpoint(completed)
                
                # Dernier batch
                if batch:
                    for r in batch:
                        f.write(r['gaps_encoded'].tobytes())
                        self.stats['total_gaps'] += r['num_gaps']
                        self.stats['total_bytes'] += r['bytes_size']
                        if self.stats['first_prime'] is None:
                            self.stats['first_prime'] = r['first_prime']
                        if r['last_prime'] is not None:
                            self.stats['last_prime'] = r['last_prime']
                    
                    self.display_progress(len(sequential_results), start_time)
            
            self.stats['end_time'] = datetime.now().isoformat()
            
        except KeyboardInterrupt:
            print("\n\n⚠ Interruption - Sauvegarde checkpoint...")
            self.save_checkpoint(self.stats['chunks_processed'])
            print("✓ Checkpoint sauvegardé")
            sys.exit(0)
        
        except Exception as e:
            print(f"\n\n❌ Erreur: {e}")
            import traceback
            traceback.print_exc()
            self.save_checkpoint(self.stats['chunks_processed'])
            raise
        
        # Supprimer checkpoint
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
        
        # Sauvegarder métadonnées
        self.save_metadata()
        
        # Stats finales
        self.display_final_stats(start_time)
    
    def display_final_stats(self, start_time: float):
        """Affiche les statistiques finales"""
        elapsed = time.time() - start_time
        
        print("\n\n" + "=" * 80)
        print("✅ GÉNÉRATION TERMINÉE")
        print("=" * 80)
        print(f"Intervalle: {self.start_number:.2e} → {self.target:.2e}")
        print(f"Total gaps: {self.stats['total_gaps']:,}")
        print(f"Premier: {self.stats['first_prime']}")
        print(f"Dernier: {self.stats['last_prime']}")
        print(f"Taille fichier: {self.format_bytes(self.stats['total_bytes'])}")
        print(f"Chunks traités: {self.stats['total_chunks']}")
        print(f"Workers: {self.num_workers}")
        print(f"Temps total: {self.format_time(elapsed)}")
        print(f"Vitesse: {(self.target - self.start_number) / elapsed:.2e} nombres/s")
        print(f"\n📁 Fichiers:")
        print(f"  • Gaps: {self.gaps_file}")
        print(f"  • Métadonnées: {self.metadata_file}")
        print("=" * 80)
    
    def save_metadata(self):
        """Sauvegarde les métadonnées"""
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
            'encoding': 'uint8 with 255 marker for larger gaps',
            'parallel_mode': True
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Métadonnées: {self.metadata_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Génération parallèle de gaps (optimisé multi-core)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  
  # Génération 10^13
  python3.13 generate_gaps_parallel_fixed.py --target 1e13 --workers 28
  
  # Génération 10^15 depuis 10^12
  python3.13 generate_gaps_parallel_fixed.py --start 1e12 --target 1e15 --workers 120
  
  # Avec buffer personnalisé
  python3.13 generate_gaps_parallel_fixed.py --target 1e15 --buffer 64 --workers 100
        """
    )
    
    parser.add_argument('--target', type=float, required=True,
                       help='Nombre cible (ex: 1e15)')
    parser.add_argument('--start', type=float,
                       help='Nombre de départ (défaut: 2)')
    parser.add_argument('--output', type=str, default='gaps_data',
                       help='Répertoire de sortie')
    parser.add_argument('--workers', type=int,
                       help='Nombre de workers (défaut: CPU-2)')
    parser.add_argument('--buffer', type=int, default=32,
                       help='Taille buffer mémoire en GB (défaut: 32)')
    
    args = parser.parse_args()
    
    # Vérifier les ressources
    cpu_count = mp.cpu_count()
    ram_gb = psutil.virtual_memory().total / 1024**3
    
    print(f"\n💻 Ressources détectées:")
    print(f"  • CPU cores: {cpu_count}")
    print(f"  • RAM totale: {ram_gb:.1f} GB")
    
    if args.workers and args.workers > cpu_count:
        print(f"⚠ Warning: --workers {args.workers} > {cpu_count} cores disponibles")
    
    if args.buffer > ram_gb * 0.8:
        print(f"⚠ Warning: Buffer {args.buffer} GB > 80% RAM ({ram_gb*0.8:.1f} GB)")
    
    print()
    
    # Créer le générateur
    generator = ParallelGapsGenerator(
        target=args.target,
        start=args.start,
        output_dir=args.output,
        num_workers=args.workers,
        buffer_size_gb=args.buffer
    )
    
    # Lancer
    generator.generate()


if __name__ == "__main__":
    # Forcer le mode 'spawn' pour éviter les problèmes de fork avec multiprocessing
    mp.set_start_method('spawn', force=True)
    main()
