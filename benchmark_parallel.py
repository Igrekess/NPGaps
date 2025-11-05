#!/usr/bin/env python3
"""
Benchmark: Compare performances séquentiel vs parallèle

Tests sur différentes échelles pour identifier le speedup optimal
"""

import subprocess
import time
import json
import psutil
from pathlib import Path


def run_benchmark(target, workers_list=[1, 2, 4, 8, 16, 32], output_dir="benchmark_results"):
    """
    Lance des benchmarks avec différents nombres de workers
    
    Args:
        target: Cible pour le test (ex: 1e10)
        workers_list: Liste du nombre de workers à tester
        output_dir: Répertoire pour les résultats
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    results = []
    
    print("=" * 80)
    print(f"🔬 BENCHMARK: Génération jusqu'à {target:.2e}")
    print("=" * 80)
    print(f"Workers à tester: {workers_list}")
    print(f"CPU disponibles: {psutil.cpu_count()}")
    print(f"RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB")
    print("=" * 80)
    print()
    
    for num_workers in workers_list:
        print(f"\n{'='*60}")
        print(f"TEST avec {num_workers} worker{'s' if num_workers > 1 else ''}")
        print(f"{'='*60}")
        
        test_output = output_path / f"test_{num_workers}workers"
        test_output.mkdir(exist_ok=True)
        
        # Lancer la génération
        start_time = time.time()
        
        cmd = [
            'python3.13',
            'generate_gaps_parallel.py',
            '--target', str(target),
            '--workers', str(num_workers),
            '--output', str(test_output)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            elapsed = time.time() - start_time
            
            if result.returncode == 0:
                # Lire les métadonnées
                metadata_file = test_output / f"metadata_to_{target:.0e}.json"
                
                if metadata_file.exists():
                    with open(metadata_file) as f:
                        metadata = json.load(f)
                    
                    result_data = {
                        'workers': num_workers,
                        'target': target,
                        'elapsed_seconds': elapsed,
                        'total_gaps': metadata['total_gaps'],
                        'total_bytes': metadata['total_bytes'],
                        'speed_numbers_per_sec': target / elapsed,
                        'success': True
                    }
                    
                    print(f"\n✅ Succès:")
                    print(f"  Temps: {elapsed:.2f} s")
                    print(f"  Vitesse: {target/elapsed:.2e} nombres/s")
                    print(f"  Gaps: {metadata['total_gaps']:,}")
                else:
                    result_data = {
                        'workers': num_workers,
                        'success': False,
                        'error': 'Metadata file not found'
                    }
                    print(f"\n❌ Échec: fichier métadonnées introuvable")
            else:
                result_data = {
                    'workers': num_workers,
                    'success': False,
                    'error': result.stderr
                }
                print(f"\n❌ Échec: {result.stderr[:200]}")
            
            results.append(result_data)
            
        except subprocess.TimeoutExpired:
            print(f"\n⏱️ Timeout dépassé (1h)")
            results.append({
                'workers': num_workers,
                'success': False,
                'error': 'Timeout'
            })
        
        except Exception as e:
            print(f"\n❌ Erreur: {e}")
            results.append({
                'workers': num_workers,
                'success': False,
                'error': str(e)
            })
    
    # Sauvegarder les résultats
    results_file = output_path / f"benchmark_{target:.0e}_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Afficher le résumé
    print("\n\n" + "=" * 80)
    print("📊 RÉSUMÉ DES BENCHMARKS")
    print("=" * 80)
    
    successful_results = [r for r in results if r.get('success')]
    
    if successful_results:
        print(f"\n{'Workers':<10} {'Temps (s)':<15} {'Vitesse':<20} {'Speedup':<10}")
        print("-" * 60)
        
        baseline_time = None
        for r in successful_results:
            if baseline_time is None:
                baseline_time = r['elapsed_seconds']
            
            speedup = baseline_time / r['elapsed_seconds'] if baseline_time else 1.0
            
            print(f"{r['workers']:<10} "
                  f"{r['elapsed_seconds']:<15.2f} "
                  f"{r['speed_numbers_per_sec']:<20.2e} "
                  f"{speedup:<10.2f}x")
        
        # Speedup optimal
        best_result = max(successful_results, key=lambda x: x.get('speed_numbers_per_sec', 0))
        optimal_speedup = baseline_time / best_result['elapsed_seconds']
        
        print(f"\n✅ Configuration optimale:")
        print(f"  Workers: {best_result['workers']}")
        print(f"  Speedup: {optimal_speedup:.2f}x")
        print(f"  Efficacité: {optimal_speedup/best_result['workers']*100:.1f}%")
    
    print(f"\n📁 Résultats sauvegardés: {results_file}")
    print("=" * 80)


def quick_test():
    """Test rapide avec 10^9 pour valider le système"""
    print("🚀 Test rapide de validation (10^9)")
    run_benchmark(
        target=1e9,
        workers_list=[1, 4, 8, 16, 32],
        output_dir="benchmark_quick"
    )


def full_benchmark():
    """Benchmark complet sur plusieurs échelles"""
    print("🔬 Benchmark complet multi-échelles")
    
    targets = [1e9, 1e10, 1e11, 1e12]
    
    for target in targets:
        print(f"\n\n{'#'*80}")
        print(f"# ÉCHELLE: {target:.0e}")
        print(f"{'#'*80}\n")
        
        run_benchmark(
            target=target,
            workers_list=[1, 8, 16, 24, 32],
            output_dir=f"benchmark_{target:.0e}"
        )


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'quick':
            quick_test()
        elif sys.argv[1] == 'full':
            full_benchmark()
        else:
            print("Usage:")
            print("  python benchmark.py quick   # Test rapide (10^9)")
            print("  python benchmark.py full    # Benchmark complet")
    else:
        print("Lancement du test rapide par défaut...")
        quick_test()
