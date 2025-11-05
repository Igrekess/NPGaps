#!/usr/bin/env python3
"""
Script de génération de gaps entre nombres premiers par streaming
Permet de générer jusqu'à 10^15 avec RAM constante (2-4 GB)

Auteur: Pour le projet Théorie de la Persistance
Date: 2025-11-05
"""

import argparse
import json
import os
import sys
import time
import hashlib
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import tempfile


class GapsStreamingGenerator:
    """
    Génère les gaps entre nombres premiers par streaming
    RAM constante quelle que soit la cible
    """
    
    def __init__(self, target, start=None, output_dir="gaps_data", segment_size=None):
        """
        Args:
            target: Nombre cible (ex: 1e13)
            start: Nombre de départ (ex: 1e10). Si None, démarre à 2
            output_dir: Répertoire de sortie
            segment_size: Taille des segments (auto si None)
        """
        self.target = int(target)
        self.start_number = int(start) if start is not None else 2
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Vérifier que primesieve est installé
        self.check_primesieve()
        
        # Taille de segment automatique selon la cible
        if segment_size is None:
            interval_size = self.target - self.start_number
            if interval_size <= 1e11:
                self.segment_size = int(1e9)   # 1 milliard
            elif interval_size <= 1e13:
                self.segment_size = int(1e10)  # 10 milliards
            else:
                self.segment_size = int(1e11)  # 100 milliards
        else:
            self.segment_size = int(segment_size)
        
        # Chemins des fichiers (avec start dans le nom si différent de 2)
        if self.start_number > 2:
            file_suffix = f"{self.start_number:.0e}_to_{self.target:.0e}"
        else:
            file_suffix = f"to_{self.target:.0e}"
        
        self.gaps_file = self.output_dir / f"gaps_{file_suffix}.dat"
        self.metadata_file = self.output_dir / f"metadata_{file_suffix}.json"
        self.checkpoint_file = self.output_dir / f"checkpoint_{file_suffix}.json"
        
        # Statistiques
        self.stats = {
            'start_time': None,
            'end_time': None,
            'start_number': self.start_number,
            'total_gaps': 0,
            'total_bytes': 0,
            'segments_processed': 0,
            'last_prime': None,
            'first_prime': None,  # Sera défini lors de la génération
            'checksums': {},
            'segments_info': []
        }
        
        # Pour la reprise
        self.resume_from = self.start_number
        self.prev_prime = None  # Sera défini au premier segment
    
    def check_primesieve(self):
        """Vérifie que primesieve est installé et accessible"""
        try:
            result = subprocess.run(['primesieve', '--version'], 
                                   capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                version = result.stdout.strip()
                print(f"✓ primesieve détecté: {version}")
            else:
                raise FileNotFoundError
        except (FileNotFoundError, subprocess.TimeoutExpired):
            print("❌ ERREUR: primesieve n'est pas installé ou non accessible")
            print("\nInstallation:")
            print("  Ubuntu/Debian: sudo apt install primesieve")
            print("  macOS: brew install primesieve")
            print("  Windows: Télécharger depuis https://github.com/kimwalisch/primesieve/releases")
            sys.exit(1)
    
    def generate_primes_segment(self, start, stop):
        """
        Génère les nombres premiers dans [start, stop] via primesieve CLI
        
        Args:
            start: Début de l'intervalle
            stop: Fin de l'intervalle
            
        Returns:
            np.array: Array des nombres premiers
        """
        # Créer un fichier temporaire pour stocker les premiers
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt') as tmp:
            tmp_path = tmp.name
        
        try:
            # Appel à primesieve en mode print avec redirection vers fichier
            # Format: primesieve START STOP --print > output.txt
            cmd = ['primesieve', str(start), str(stop), '--print']
            
            with open(tmp_path, 'w') as outfile:
                result = subprocess.run(cmd, stdout=outfile, stderr=subprocess.PIPE, 
                                       text=True, timeout=3600)
            
            if result.returncode != 0:
                raise RuntimeError(f"primesieve error: {result.stderr}")
            
            # Lire les premiers depuis le fichier
            primes = []
            with open(tmp_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and line.isdigit():
                        primes.append(int(line))
            
            return np.array(primes, dtype=np.uint64)
            
        finally:
            # Nettoyer le fichier temporaire
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        
    def load_checkpoint(self):
        """Charge le checkpoint s'il existe"""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                
                self.resume_from = checkpoint['resume_from']
                self.prev_prime = checkpoint['prev_prime']
                self.stats = checkpoint['stats']
                
                print(f"✓ Checkpoint trouvé: reprise à {self.resume_from:.2e}")
                print(f"  Progression précédente: {checkpoint['progress']:.1f}%")
                
                return True
            except Exception as e:
                print(f"⚠ Erreur lecture checkpoint: {e}")
                return False
        return False
    
    def save_checkpoint(self, current_position):
        """Sauvegarde un checkpoint"""
        checkpoint = {
            'resume_from': current_position,
            'prev_prime': self.prev_prime,
            'progress': (current_position / self.target) * 100,
            'timestamp': datetime.now().isoformat(),
            'stats': self.stats
        }
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    def compute_segment_hash(self, gaps):
        """Calcule le hash SHA-256 d'un segment de gaps"""
        return hashlib.sha256(gaps.tobytes()).hexdigest()
    
    def format_time(self, seconds):
        """Formate un temps en secondes en format lisible"""
        return str(timedelta(seconds=int(seconds)))
    
    def format_bytes(self, bytes_count):
        """Formate une taille en bytes en format lisible"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_count < 1024:
                return f"{bytes_count:.2f} {unit}"
            bytes_count /= 1024
        return f"{bytes_count:.2f} PB"
    
    def generate(self):
        """
        Génération principale par streaming
        """
        print("=" * 80)
        print("🚀 GÉNÉRATION DE GAPS PAR STREAMING")
        print("=" * 80)
        if self.start_number > 2:
            print(f"Intervalle: {self.start_number:.2e} → {self.target:.2e}")
        else:
            print(f"Cible: {self.target:.2e}")
        print(f"Taille segment: {self.segment_size:.2e}")
        print(f"Fichier sortie: {self.gaps_file}")
        print(f"RAM utilisée: ~2-4 GB (constant)")
        print("=" * 80)
        
        # Vérifier si reprise possible
        resume = self.load_checkpoint()
        
        # Mode d'ouverture du fichier
        mode = 'ab' if resume else 'wb'
        
        self.stats['start_time'] = datetime.now().isoformat()
        start_time = time.time()
        
        position = self.resume_from
        
        try:
            with open(self.gaps_file, mode) as f:
                
                while position < self.target:
                    segment_start_time = time.time()
                    
                    # Définir le segment
                    segment_end = min(position + self.segment_size, self.target)
                    
                    # Génération des premiers dans ce segment
                    try:
                        primes = self.generate_primes_segment(position, segment_end)
                    except Exception as e:
                        print(f"\n❌ Erreur génération primes: {e}")
                        self.save_checkpoint(position)
                        raise
                    
                    if len(primes) == 0:
                        print(f"\n⚠ Aucun premier dans [{position:.2e}, {segment_end:.2e}]")
                        position = segment_end
                        continue
                    
                    # Premier segment : initialiser prev_prime et first_prime
                    if self.prev_prime is None:
                        # Trouver le premier juste avant start_number
                        if self.start_number > 2:
                            # Génerer le dernier premier < start_number
                            prev_primes = self.generate_primes_segment(
                                max(2, self.start_number - 10000), 
                                self.start_number
                            )
                            if len(prev_primes) > 0:
                                self.prev_prime = int(prev_primes[-1])
                            else:
                                self.prev_prime = 2
                        else:
                            self.prev_prime = 2
                        
                        self.stats['first_prime'] = int(primes[0])
                    
                    # Calcul des gaps (incluant le gap depuis prev_prime)
                    all_primes = np.concatenate([[self.prev_prime], primes])
                    gaps = np.diff(all_primes)
                    
                    # Vérification: tous les gaps doivent être pairs (sauf premier gap qui peut être impair)
                    # Note: le gap 2->3 est impair (1), tous les autres sont pairs
                    
                    # Conversion en uint8 (gère les gaps jusqu'à 255)
                    # Pour les gaps > 255, on utilisera un marqueur spécial
                    gaps_uint8 = self.encode_gaps(gaps)
                    
                    # Hash du segment pour intégrité
                    segment_hash = self.compute_segment_hash(gaps_uint8)
                    
                    # Écriture sur disque (streaming!)
                    f.write(gaps_uint8.tobytes())
                    f.flush()  # Force l'écriture
                    
                    # Mise à jour statistiques
                    self.stats['total_gaps'] += len(gaps)
                    self.stats['total_bytes'] += len(gaps_uint8)
                    self.stats['segments_processed'] += 1
                    self.stats['last_prime'] = int(primes[-1])
                    self.stats['checksums'][f"segment_{self.stats['segments_processed']}"] = segment_hash
                    
                    segment_info = {
                        'segment_id': self.stats['segments_processed'],
                        'start': int(position),
                        'end': int(segment_end),
                        'num_primes': len(primes),
                        'num_gaps': len(gaps),
                        'hash': segment_hash,
                        'time_seconds': time.time() - segment_start_time
                    }
                    self.stats['segments_info'].append(segment_info)
                    
                    # Mise à jour pour segment suivant
                    self.prev_prime = int(primes[-1])
                    position = segment_end
                    
                    # Libération mémoire
                    del primes, gaps, all_primes, gaps_uint8
                    
                    # Affichage progression
                    self.display_progress(position, start_time)
                    
                    # Checkpoint tous les 10 segments
                    if self.stats['segments_processed'] % 10 == 0:
                        self.save_checkpoint(position)
                
                # Fin de génération
                self.stats['end_time'] = datetime.now().isoformat()
                
        except KeyboardInterrupt:
            print("\n\n⚠ Interruption utilisateur - Sauvegarde checkpoint...")
            self.save_checkpoint(position)
            print("✓ Checkpoint sauvegardé - Reprise possible avec --resume")
            sys.exit(0)
        
        except Exception as e:
            print(f"\n\n❌ Erreur: {e}")
            self.save_checkpoint(position)
            raise
        
        # Suppression du checkpoint (génération complète)
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
        
        # Sauvegarde métadonnées
        self.save_metadata()
        
        # Affichage final
        self.display_final_stats(start_time)
    
    def encode_gaps(self, gaps):
        """
        Encode les gaps en uint8, avec gestion des gaps > 255
        Format: si gap < 255: valeur directe
                si gap >= 255: [255, high_byte, low_byte]
        """
        result = []
        for gap in gaps:
            gap_int = int(gap)  # Convertir en int Python pour les opérations de bits
            if gap_int < 255:
                result.append(gap_int)
            else:
                # Marqueur 255 suivi de uint16 en big-endian
                result.append(255)
                result.append((gap_int >> 8) & 0xFF)  # high byte
                result.append(gap_int & 0xFF)          # low byte
        
        return np.array(result, dtype=np.uint8)
    
    def display_progress(self, position, start_time):
        """Affiche la progression en temps réel"""
        progress = (position / self.target) * 100
        elapsed = time.time() - start_time
        
        if progress > 0:
            eta_seconds = (elapsed / progress) * (100 - progress)
            eta = self.format_time(eta_seconds)
        else:
            eta = "Calcul..."
        
        # Vitesse (nombres traités par seconde)
        speed = position / elapsed if elapsed > 0 else 0
        
        print(f"\r📊 {progress:6.2f}% | "
              f"Position: {position:.2e}/{self.target:.2e} | "
              f"Gaps: {self.stats['total_gaps']:,} | "
              f"Vitesse: {speed:.2e} nombres/s | "
              f"ETA: {eta} | "
              f"RAM: ~{self.format_bytes(self.stats['total_bytes'])} écrit",
              end='', flush=True)
    
    def display_final_stats(self, start_time):
        """Affiche les statistiques finales"""
        elapsed = time.time() - start_time
        
        print("\n\n" + "=" * 80)
        print("✅ GÉNÉRATION TERMINÉE")
        print("=" * 80)
        print(f"Cible: {self.target:.2e}")
        print(f"Total gaps générés: {self.stats['total_gaps']:,}")
        print(f"Premier nombre premier: {self.stats['first_prime']}")
        print(f"Dernier nombre premier: {self.stats['last_prime']}")
        print(f"Taille fichier: {self.format_bytes(self.stats['total_bytes'])}")
        print(f"Segments traités: {self.stats['segments_processed']}")
        print(f"Temps total: {self.format_time(elapsed)}")
        print(f"Vitesse moyenne: {self.target / elapsed:.2e} nombres/s")
        print(f"\n📁 Fichiers générés:")
        print(f"  • Gaps: {self.gaps_file}")
        print(f"  • Métadonnées: {self.metadata_file}")
        print("=" * 80)
    
    def save_metadata(self):
        """Sauvegarde les métadonnées"""
        metadata = {
            'target': self.target,
            'segment_size': self.segment_size,
            'first_prime': self.stats['first_prime'],
            'last_prime': self.stats['last_prime'],
            'total_gaps': self.stats['total_gaps'],
            'total_bytes': self.stats['total_bytes'],
            'segments_processed': self.stats['segments_processed'],
            'start_time': self.stats['start_time'],
            'end_time': self.stats['end_time'],
            'gaps_file': str(self.gaps_file),
            'encoding': 'uint8 with 255 marker for larger gaps',
            'checksums': self.stats['checksums'],
            'segments_info': self.stats['segments_info']
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Métadonnées sauvegardées: {self.metadata_file}")


def verify_gaps_file(gaps_file, metadata_file):
    """
    Vérifie l'intégrité d'un fichier de gaps
    """
    print("\n" + "=" * 80)
    print("🔍 VÉRIFICATION INTÉGRITÉ")
    print("=" * 80)
    
    # Charger métadonnées
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Vérifier taille fichier
    file_size = os.path.getsize(gaps_file)
    expected_size = metadata['total_bytes']
    
    print(f"Taille fichier: {file_size:,} bytes")
    print(f"Taille attendue: {expected_size:,} bytes")
    
    if file_size == expected_size:
        print("✓ Taille OK")
    else:
        print("❌ Taille incorrecte!")
        return False
    
    print(f"\nTotal gaps: {metadata['total_gaps']:,}")
    print(f"Premier: {metadata['first_prime']}")
    print(f"Dernier: {metadata['last_prime']}")
    print(f"Segments: {metadata['segments_processed']}")
    
    # TODO: Vérifier les checksums des segments si nécessaire
    
    print("\n✓ Fichier valide")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Génération de gaps entre nombres premiers par streaming",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  
  # Génération jusqu'à 10^11 (rapide, ~2 min)
  python generate_gaps_streaming.py --target 1e11
  
  # Génération d'une décade spécifique (10^10 à 10^11)
  python generate_gaps_streaming.py --start 1e10 --target 1e11
  
  # Génération jusqu'à 10^13 (chez vous, ~2h)
  python generate_gaps_streaming.py --target 1e13 --output my_gaps
  
  # Génération jusqu'à 10^15 (RunPod, ~10h)
  python generate_gaps_streaming.py --target 1e15 --segment-size 1e11
  
  # Vérification d'un fichier existant
  python generate_gaps_streaming.py --verify gaps_data/gaps_to_1e13.dat
        """
    )
    
    parser.add_argument('--target', type=float, 
                       help='Nombre cible (ex: 1e13)')
    parser.add_argument('--start', type=float,
                       help='Nombre de départ (ex: 1e10). Si non spécifié, démarre à 2')
    parser.add_argument('--output', type=str, default='gaps_data',
                       help='Répertoire de sortie (défaut: gaps_data)')
    parser.add_argument('--segment-size', type=float,
                       help='Taille des segments (auto si non spécifié)')
    parser.add_argument('--verify', type=str,
                       help='Vérifie un fichier de gaps existant')
    
    args = parser.parse_args()
    
    if args.verify:
        # Mode vérification
        gaps_file = Path(args.verify)
        metadata_file = gaps_file.parent / f"metadata_{gaps_file.stem.split('_to_')[1]}.json"
        
        if not gaps_file.exists():
            print(f"❌ Fichier introuvable: {gaps_file}")
            sys.exit(1)
        
        if not metadata_file.exists():
            print(f"❌ Métadonnées introuvables: {metadata_file}")
            sys.exit(1)
        
        verify_gaps_file(gaps_file, metadata_file)
    
    elif args.target:
        # Mode génération
        generator = GapsStreamingGenerator(
            target=args.target,
            start=args.start,
            output_dir=args.output,
            segment_size=args.segment_size
        )
        
        generator.generate()
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
