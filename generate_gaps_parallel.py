#!/usr/bin/env python3
"""
Générateur de gaps parallèle - Version robuste
Corrections: communication inter-processus + gestion erreurs pickle
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
import traceback as tb


def generate_primes_chunk(start: int, stop: int) -> np.ndarray:
    """Génère les nombres premiers"""
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt') as tmp:
        tmp_path = tmp.name
    
    try:
        cmd = ['primesieve', str(start), str(stop), '--print']
        with open(tmp_path, 'w') as outfile:
            with open(os.devnull, 'w') as devnull:
                result = subprocess.run(cmd, stdout=outfile, stderr=devnull, timeout=3600)
        
        if result.returncode != 0:
            return np.array([], dtype=np.uint64)
        
        primes = []
        with open(tmp_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and line.isdigit():
                    primes.append(int(line))
        
        return np.array(primes, dtype=np.uint64)
    except:
        return np.array([], dtype=np.uint64)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except:
            pass


def encode_gaps(gaps: np.ndarray) -> bytes:
    """Encode gaps en bytes directement"""
    result = bytearray()
    for gap in gaps:
        gap_int = int(gap)
        if gap_int < 255:
            result.append(gap_int)
        else:
            result.extend([255, (gap_int >> 8) & 0xFF, gap_int & 0xFF])
    return bytes(result)


def process_chunk_worker(args: Tuple[int, int, int]) -> Dict:
    """Worker - retourne seulement le premier et dernier premier"""
    chunk_id, start, stop = args
    
    try:
        primes = generate_primes_chunk(start, stop)
        
        if len(primes) == 0:
            return {
                'id': chunk_id,
                'ok': True,
                'n': 0,
                'first': None,
                'last': None
            }
        
        # Ne retourner que le premier et dernier (économie mémoire)
        return {
            'id': chunk_id,
            'ok': True,
            'n': len(primes),
            'first': int(primes[0]),
            'last': int(primes[-1]),
            'start': start,
            'stop': stop
        }
    except Exception as e:
        return {
            'id': chunk_id,
            'ok': False,
            'error': str(e)
        }


class ParallelGapsGenerator:
    
    def __init__(self, target, start=None, output_dir="gaps_data", 
                 num_workers=None, buffer_size_gb=32):
        self.target = int(target)
        self.start_number = int(start) if start is not None else 2
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        cpu_count = mp.cpu_count()
        self.num_workers = min(num_workers if num_workers else cpu_count - 2, cpu_count)
        
        self.check_primesieve()
        
        # Chunks plus gros pour réduire overhead
        interval_size = self.target - self.start_number
        self.chunk_size = max(int(interval_size / (self.num_workers * 50)), int(1e10))
        
        if self.chunk_size >= 1e11:
            self.chunk_size = int(np.round(self.chunk_size / 1e11) * 1e11)
        elif self.chunk_size >= 1e10:
            self.chunk_size = int(np.round(self.chunk_size / 1e10) * 1e10)
        
        if self.start_number > 2:
            file_suffix = f"{self.start_number:.0e}_to_{self.target:.0e}"
        else:
            file_suffix = f"to_{self.target:.0e}"
        
        self.gaps_file = self.output_dir / f"gaps_{file_suffix}.dat"
        self.metadata_file = self.output_dir / f"metadata_{file_suffix}.json"
        
        self.stats = {
            'start_time': None,
            'total_gaps': 0,
            'total_bytes': 0,
            'first_prime': None,
            'last_prime': None
        }
    
    def check_primesieve(self):
        try:
            subprocess.run(['primesieve', '--version'], 
                          capture_output=True, timeout=5, check=True)
        except:
            print("❌ primesieve non installé")
            sys.exit(1)
    
    def prepare_chunks(self):
        chunks = []
        pos = self.start_number
        chunk_id = 0
        
        while pos < self.target:
            stop = min(pos + self.chunk_size, self.target)
            chunks.append((chunk_id, pos, stop))
            pos = stop
            chunk_id += 1
        
        return chunks
    
    def generate(self):
        print("=" * 60)
        print(f"Intervalle: {self.start_number:.2e} → {self.target:.2e}")
        print(f"Workers: {self.num_workers}")
        print(f"Chunk: {self.chunk_size:.2e}")
        print("=" * 60)
        
        chunks = self.prepare_chunks()
        print(f"\n📦 {len(chunks)} chunks\n")
        
        self.stats['start_time'] = datetime.now().isoformat()
        start_time = time.time()
        
        # Phase 1: Collecter infos chunks
        print("Phase 1/2: Scan chunks...")
        chunk_infos = []
        
        try:
            with Pool(processes=self.num_workers, maxtasksperchild=10) as pool:
                for i, result in enumerate(pool.imap(process_chunk_worker, chunks, chunksize=1)):
                    chunk_infos.append(result)
                    
                    if (i + 1) % 50 == 0 or (i + 1) == len(chunks):
                        progress = ((i + 1) / len(chunks)) * 100
                        print(f"\r  {i+1}/{len(chunks)} ({progress:5.1f}%)", end='', flush=True)
            
            print(f"\r  ✅ {len(chunk_infos)} chunks scannés\n")
        except Exception as e:
            print(f"\n❌ Erreur phase 1: {e}")
            tb.print_exc()
            return
        
        # Trier par ID
        chunk_infos.sort(key=lambda x: x['id'])
        
        # Phase 2: Générer et écrire gaps
        print("Phase 2/2: Génération gaps...")
        
        # Trouver prev_prime initial
        prev_prime = None
        if self.start_number > 2:
            search_start = max(2, self.start_number - 1000)
            primes_before = generate_primes_chunk(search_start, self.start_number)
            if len(primes_before) > 0:
                prev_prime = int(primes_before[-1])
        
        with open(self.gaps_file, 'wb') as f:
            for i, info in enumerate(chunk_infos):
                if not info['ok'] or info['n'] == 0:
                    continue
                
                # Régénérer les premiers du chunk
                primes = generate_primes_chunk(info['start'], info['stop'])
                
                if len(primes) == 0:
                    continue
                
                # Calculer gaps
                if prev_prime is not None:
                    all_primes = np.concatenate([[prev_prime], primes])
                    gaps = np.diff(all_primes)
                else:
                    gaps = np.diff(primes)
                
                # Valider
                if np.any(gaps <= 0):
                    print(f"\n⚠️  Chunk {info['id']}: gaps invalides, skip")
                    continue
                
                # Encoder et écrire
                gaps_bytes = encode_gaps(gaps)
                f.write(gaps_bytes)
                
                self.stats['total_gaps'] += len(gaps)
                self.stats['total_bytes'] += len(gaps_bytes)
                
                if self.stats['first_prime'] is None:
                    self.stats['first_prime'] = int(primes[0])
                self.stats['last_prime'] = int(primes[-1])
                
                prev_prime = int(primes[-1])
                
                if (i + 1) % 50 == 0 or (i + 1) == len(chunk_infos):
                    progress = ((i + 1) / len(chunk_infos)) * 100
                    print(f"\r  {i+1}/{len(chunk_infos)} ({progress:5.1f}%) | "
                          f"{self.stats['total_bytes'] / 1024**3:.1f} GB",
                          end='', flush=True)
        
        print(f"\r  ✅ {len(chunk_infos)} chunks écrits | "
              f"{self.stats['total_bytes'] / 1024**3:.1f} GB\n")
        
        elapsed = time.time() - start_time
        
        # Stats finales
        print("=" * 60)
        print(f"✅ TERMINÉ")
        print(f"Gaps: {self.stats['total_gaps']:,}")
        print(f"Premier: {self.stats['first_prime']}")
        print(f"Dernier: {self.stats['last_prime']}")
        print(f"Taille: {self.stats['total_bytes'] / 1024**3:.1f} GB")
        print(f"Temps: {str(timedelta(seconds=int(elapsed)))}")
        print(f"Vitesse: {(self.target - self.start_number) / elapsed:.2e} n/s")
        print("=" * 60)
        
        # Métadonnées
        self.stats['end_time'] = datetime.now().isoformat()
        metadata = {
            'target': self.target,
            'start_number': self.start_number,
            'num_workers': self.num_workers,
            'first_prime': self.stats['first_prime'],
            'last_prime': self.stats['last_prime'],
            'total_gaps': self.stats['total_gaps'],
            'total_bytes': self.stats['total_bytes'],
            'start_time': self.stats['start_time'],
            'end_time': self.stats['end_time']
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=float, required=True)
    parser.add_argument('--start', type=float)
    parser.add_argument('--output', type=str, default='gaps_data')
    parser.add_argument('--workers', type=int)
    parser.add_argument('--buffer', type=int, default=32)
    
    args = parser.parse_args()
    
    cpu = mp.cpu_count()
    ram = psutil.virtual_memory().total / 1024**3
    
    print(f"\n💻 CPU: {cpu} | RAM: {ram:.1f} GB\n")
    
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
