#!/usr/bin/env python3
"""
Générateur de gaps parallèle avec monitoring complet
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
from typing import Tuple, Dict
import psutil


def generate_primes_chunk(start: int, stop: int) -> np.ndarray:
    """Génère les nombres premiers"""
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        cmd = ['primesieve', str(start), str(stop), '--print']
        with open(tmp_path, 'w') as outfile:
            subprocess.run(cmd, stdout=outfile, stderr=subprocess.DEVNULL, timeout=3600)
        
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
            os.unlink(tmp_path)
        except:
            pass


def encode_gaps(gaps: np.ndarray) -> bytes:
    """Encode gaps en bytes"""
    result = bytearray()
    for gap in gaps:
        g = int(gap)
        if g < 255:
            result.append(g)
        else:
            result.extend([255, (g >> 8) & 0xFF, g & 0xFF])
    return bytes(result)


def process_chunk(args: Tuple[int, int, int]) -> Dict:
    """Worker"""
    chunk_id, start, stop = args
    
    try:
        primes = generate_primes_chunk(start, stop)
        
        if len(primes) == 0:
            return {'id': chunk_id, 'ok': True, 'n': 0, 'first': None, 'last': None}
        
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
        return {'id': chunk_id, 'ok': False, 'error': str(e)}


class Generator:
    
    def __init__(self, target, start=None, output_dir="gaps_data", num_workers=None):
        self.target = int(target)
        self.start_number = int(start) if start else 2
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        cpu = mp.cpu_count()
        self.num_workers = min(num_workers if num_workers else cpu - 2, cpu)
        
        # Chunk size
        interval = self.target - self.start_number
        self.chunk_size = max(int(interval / (self.num_workers * 50)), int(1e10))
        
        if self.chunk_size >= 1e11:
            self.chunk_size = int(np.round(self.chunk_size / 1e11) * 1e11)
        else:
            self.chunk_size = int(np.round(self.chunk_size / 1e10) * 1e10)
        
        # Fichiers
        suffix = f"{self.start_number:.0e}_to_{self.target:.0e}" if self.start_number > 2 else f"to_{self.target:.0e}"
        self.gaps_file = self.output_dir / f"gaps_{suffix}.dat"
        self.metadata_file = self.output_dir / f"metadata_{suffix}.json"
        
        self.stats = {'total_gaps': 0, 'total_bytes': 0, 'first_prime': None, 'last_prime': None}
    
    def prepare_chunks(self):
        chunks = []
        pos = self.start_number
        cid = 0
        
        while pos < self.target:
            stop = min(pos + self.chunk_size, self.target)
            chunks.append((cid, pos, stop))
            pos = stop
            cid += 1
        
        return chunks
    
    def run(self):
        print("=" * 80)
        print(f"Intervalle: {self.start_number:.2e} → {self.target:.2e}")
        print(f"Workers: {self.num_workers} | Chunk: {self.chunk_size:.2e}")
        print("=" * 80)
        
        chunks = self.prepare_chunks()
        print(f"\n📦 {len(chunks)} chunks\n")
        
        start_time = time.time()
        
        # Phase 1: Scan chunks
        print("Phase 1/2: Scan chunks...")
        chunk_infos = []
        last_update = time.time()
        
        with Pool(self.num_workers, maxtasksperchild=10) as pool:
            for i, res in enumerate(pool.imap(process_chunk, chunks, chunksize=1)):
                chunk_infos.append(res)
                
                current_time = time.time()
                if (current_time - last_update) >= 1.0 or (i + 1) == len(chunks):
                    pct = ((i + 1) / len(chunks)) * 100
                    print(f"\r  {i+1}/{len(chunks)} ({pct:5.1f}%)", end='', flush=True)
                    last_update = current_time
        
        print(f"\r  ✅ {len(chunk_infos)} chunks scannés\n")
        
        # Trier
        chunk_infos.sort(key=lambda x: x['id'])
        
        # Phase 2: Génération gaps avec monitoring complet
        print("Phase 2/2: Génération gaps...")
        phase2_start = time.time()
        
        # Premier précédent si start > 2
        prev_prime = None
        if self.start_number > 2:
            search_start = max(2, self.start_number - 1000)
            primes_before = generate_primes_chunk(search_start, self.start_number)
            if len(primes_before) > 0:
                prev_prime = int(primes_before[-1])
        
        current_position = self.start_number
        last_update = time.time()
        
        with open(self.gaps_file, 'wb') as f:
            for i, info in enumerate(chunk_infos):
                if not info['ok'] or info['n'] == 0:
                    continue
                
                # Régénérer les premiers
                primes = generate_primes_chunk(info['start'], info['stop'])
                
                if len(primes) == 0:
                    continue
                
                # Gaps
                if prev_prime is not None:
                    all_primes = np.concatenate([[prev_prime], primes])
                    gaps = np.diff(all_primes)
                else:
                    gaps = np.diff(primes)
                
                # Validation
                if np.any(gaps <= 0):
                    print(f"\n⚠️  Chunk {info['id']}: gaps invalides")
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
                current_position = info['stop']
                
                # Affichage détaillé (mise à jour toutes les secondes)
                current_time = time.time()
                if (current_time - last_update) >= 1.0 or (i + 1) == len(chunk_infos):
                    elapsed = current_time - phase2_start
                    progress_pct = (current_position / self.target) * 100
                    
                    # Vitesse
                    speed = (current_position - self.start_number) / elapsed if elapsed > 0 else 0
                    
                    # ETA
                    if speed > 0:
                        remaining = self.target - current_position
                        eta_seconds = remaining / speed
                        eta_str = str(timedelta(seconds=int(eta_seconds)))
                    else:
                        eta_str = "..."
                    
                    # Taille
                    size_gb = self.stats['total_bytes'] / 1024**3
                    
                    print(f"\r📊 {progress_pct:6.2f}% | "
                          f"Pos: {current_position:.2e}/{self.target:.2e} | "
                          f"Gaps: {self.stats['total_gaps']:,} | "
                          f"Vitesse: {speed:.2e} n/s | "
                          f"ETA: {eta_str:>8} | "
                          f"Écrit: {size_gb:.2f} GB",
                          end='', flush=True)
                    
                    last_update = current_time
        
        print(f"\r✅ Phase 2 terminée | "
              f"Gaps: {self.stats['total_gaps']:,} | "
              f"Taille: {self.stats['total_bytes'] / 1024**3:.2f} GB" + " " * 20)
        
        elapsed = time.time() - start_time
        
        print("\n" + "=" * 80)
        print(f"✅ TERMINÉ en {str(timedelta(seconds=int(elapsed)))}")
        print(f"Gaps: {self.stats['total_gaps']:,}")
        print(f"Premier: {self.stats['first_prime']}")
        print(f"Dernier: {self.stats['last_prime']}")
        print(f"Taille: {self.stats['total_bytes'] / 1024**3:.1f} GB")
        print(f"Vitesse: {(self.target - self.start_number) / elapsed:.2e} n/s")
        print("=" * 80)
        
        # Metadata
        with open(self.metadata_file, 'w') as f:
            json.dump({
                'target': self.target,
                'start': self.start_number,
                'workers': self.num_workers,
                'first_prime': self.stats['first_prime'],
                'last_prime': self.stats['last_prime'],
                'total_gaps': self.stats['total_gaps'],
                'total_bytes': self.stats['total_bytes'],
                'time_seconds': int(elapsed)
            }, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=float, required=True)
    parser.add_argument('--start', type=float)
    parser.add_argument('--output', default='gaps_data')
    parser.add_argument('--workers', type=int)
    
    args = parser.parse_args()
    
    cpu = mp.cpu_count()
    ram = psutil.virtual_memory().total / 1024**3
    
    print(f"\n💻 CPU: {cpu} | RAM: {ram:.0f} GB\n")
    
    gen = Generator(args.target, args.start, args.output, args.workers)
    gen.run()


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
