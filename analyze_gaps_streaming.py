#!/usr/bin/env python3
"""
Script d'analyse des gaps générés par streaming
Permet d'analyser les gaps sans tout charger en RAM

Auteur: Pour le projet Théorie de la Persistance
Date: 2025-11-05
"""

import argparse
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


class GapsAnalyzer:
    """
    Analyse les gaps générés par streaming
    """
    
    def __init__(self, gaps_file, metadata_file=None):
        self.gaps_file = Path(gaps_file)
        
        # Trouver le fichier de métadonnées
        if metadata_file:
            self.metadata_file = Path(metadata_file)
        else:
            # Déduire depuis le nom du fichier
            target = self.gaps_file.stem.split('_to_')[1]
            self.metadata_file = self.gaps_file.parent / f"metadata_{target}.json"
        
        # Charger métadonnées
        with open(self.metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        print(f"✓ Fichier: {self.gaps_file}")
        print(f"✓ Cible: {self.metadata['target']:.2e}")
        print(f"✓ Total gaps: {self.metadata['total_gaps']:,}")
    
    def decode_gaps_streaming(self, chunk_size=10000000):
        """
        Générateur qui lit et décode les gaps par chunks
        Permet l'analyse sans charger tout en RAM
        
        Yields:
            np.array: Chunk de gaps décodés
        """
        with open(self.gaps_file, 'rb') as f:
            buffer = []
            
            while True:
                # Lire un chunk de bytes
                chunk = f.read(chunk_size)
                if not chunk:
                    # Fin du fichier
                    if buffer:
                        yield np.array(buffer, dtype=np.uint16)
                    break
                
                # Convertir en array
                bytes_array = np.frombuffer(chunk, dtype=np.uint8)
                
                # Décoder
                i = 0
                while i < len(bytes_array):
                    if bytes_array[i] < 255:
                        # Gap normal
                        buffer.append(bytes_array[i])
                        i += 1
                    else:
                        # Gap encodé sur 3 bytes
                        if i + 2 < len(bytes_array):
                            high = bytes_array[i + 1]
                            low = bytes_array[i + 2]
                            gap = (high << 8) | low
                            buffer.append(gap)
                            i += 3
                        else:
                            # Cas limite: gap à cheval sur deux chunks
                            # On le traite au prochain chunk
                            break
                    
                    # Yield quand le buffer est plein
                    if len(buffer) >= chunk_size // 2:
                        yield np.array(buffer, dtype=np.uint16)
                        buffer = []
    
    def compute_statistics(self, max_gaps=None):
        """
        Calcule les statistiques de base sur les gaps
        
        Args:
            max_gaps: Nombre maximum de gaps à analyser (None = tous)
        """
        print("\n" + "=" * 80)
        print("📊 STATISTIQUES DES GAPS")
        print("=" * 80)
        
        # Statistiques en streaming
        count = 0
        sum_gaps = 0
        sum_squares = 0
        min_gap = float('inf')
        max_gap = 0
        
        # Distribution
        gap_counts = {}
        
        for chunk in self.decode_gaps_streaming():
            for gap in chunk:
                count += 1
                sum_gaps += gap
                sum_squares += gap * gap
                min_gap = min(min_gap, gap)
                max_gap = max(max_gap, gap)
                
                # Comptage pour distribution
                gap_counts[gap] = gap_counts.get(gap, 0) + 1
                
                if max_gaps and count >= max_gaps:
                    break
            
            if max_gaps and count >= max_gaps:
                break
        
        # Calcul des statistiques
        mean = sum_gaps / count if count > 0 else 0
        variance = (sum_squares / count - mean * mean) if count > 0 else 0
        std = np.sqrt(variance)
        
        print(f"\nNombre de gaps analysés: {count:,}")
        print(f"\nGap minimum: {min_gap}")
        print(f"Gap maximum: {max_gap}")
        print(f"Gap moyen: {mean:.2f}")
        print(f"Écart-type: {std:.2f}")
        
        # Gaps les plus fréquents
        print(f"\n🔢 Top 10 gaps les plus fréquents:")
        sorted_gaps = sorted(gap_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        for gap, freq in sorted_gaps:
            percentage = (freq / count) * 100
            print(f"  Gap {gap:3d}: {freq:12,} occurrences ({percentage:5.2f}%)")
        
        return {
            'count': count,
            'min': min_gap,
            'max': max_gap,
            'mean': mean,
            'std': std,
            'gap_distribution': gap_counts
        }
    
    def compute_persistence_index(self, p=2, max_samples=1000000):
        """
        Calcule l'indice de persistance I(p,N) pour un premier p donné
        Utilise l'espace Z/(2p)Z pour la projection modulaire
        
        Args:
            p: Nombre premier pour la projection
            max_samples: Nombre max d'échantillons (pour rapidité)
        
        Returns:
            dict: I(p,N) et statistiques associées
        """
        print(f"\n" + "=" * 80)
        print(f"🧮 CALCUL INDICE DE PERSISTANCE I({p}, N)")
        print("=" * 80)
        
        # Pour calculer I(p,N), on regarde la distribution des gaps mod (2p)
        # L'information mutuelle mesure la dépendance
        
        modulo = 2 * p
        
        # Compter les occurrences dans chaque classe
        class_counts = np.zeros(modulo, dtype=np.int64)
        total_gaps = 0
        
        for chunk in self.decode_gaps_streaming():
            gaps_sample = chunk[:max_samples - total_gaps] if total_gaps + len(chunk) > max_samples else chunk
            
            for gap in gaps_sample:
                class_counts[gap % modulo] += 1
                total_gaps += 1
                
                if total_gaps >= max_samples:
                    break
            
            if total_gaps >= max_samples:
                break
        
        # Calcul de l'entropie de Shannon
        probabilities = class_counts / total_gaps
        probabilities = probabilities[probabilities > 0]  # Éviter log(0)
        
        entropy = -np.sum(probabilities * np.log2(probabilities))
        max_entropy = np.log2(modulo)  # Entropie maximale (distribution uniforme)
        
        # L'indice de persistance mesure la déviation par rapport à l'uniforme
        I_p = max_entropy - entropy
        
        print(f"\nÉchantillon: {total_gaps:,} gaps")
        print(f"Espace de projection: Z/({modulo})Z")
        print(f"Entropie mesurée: {entropy:.4f} bits")
        print(f"Entropie maximale: {max_entropy:.4f} bits")
        print(f"I({p}, N) = {I_p:.6f} bits")
        print(f"Taux de concentration: {(I_p / max_entropy * 100):.2f}%")
        
        # Classes les plus représentées
        top_classes = np.argsort(class_counts)[::-1][:5]
        print(f"\n🎯 Top 5 classes modulaires les plus fréquentes:")
        for cls in top_classes:
            if class_counts[cls] > 0:
                freq = class_counts[cls]
                percentage = (freq / total_gaps) * 100
                print(f"  Classe {cls:3d} (mod {modulo}): {freq:10,} ({percentage:5.2f}%)")
        
        return {
            'p': p,
            'modulo': modulo,
            'I_p': I_p,
            'entropy': entropy,
            'max_entropy': max_entropy,
            'concentration_rate': I_p / max_entropy,
            'samples': total_gaps,
            'class_distribution': class_counts.tolist()
        }
    
    def plot_gap_distribution(self, max_gap=100, output_file=None):
        """
        Crée un histogramme de la distribution des gaps
        
        Args:
            max_gap: Gap maximum à afficher
            output_file: Fichier de sortie (None = affichage)
        """
        print("\n📈 Génération du graphique...")
        
        # Compter les gaps jusqu'à max_gap
        gap_counts = np.zeros(max_gap + 1, dtype=np.int64)
        total_gaps = 0
        
        for chunk in self.decode_gaps_streaming():
            for gap in chunk:
                if gap <= max_gap:
                    gap_counts[gap] += 1
                total_gaps += 1
        
        # Créer le graphique
        fig, ax = plt.subplots(figsize=(12, 6))
        
        gaps = np.arange(max_gap + 1)
        frequencies = gap_counts / total_gaps * 100
        
        ax.bar(gaps, frequencies, width=1, edgecolor='none', alpha=0.7)
        ax.set_xlabel('Gap', fontsize=12)
        ax.set_ylabel('Fréquence (%)', fontsize=12)
        ax.set_title(f'Distribution des gaps (cible: {self.metadata["target"]:.2e})', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Annoter les gaps les plus fréquents
        top_gaps = np.argsort(gap_counts)[::-1][:5]
        for gap in top_gaps:
            if gap <= max_gap and gap_counts[gap] > 0:
                ax.annotate(f'{gap}', 
                           xy=(gap, frequencies[gap]),
                           xytext=(0, 5), textcoords='offset points',
                           ha='center', fontsize=8)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=150)
            print(f"✓ Graphique sauvegardé: {output_file}")
        else:
            plt.show()
    
    def sample_gaps(self, n_samples=1000, output_file=None):
        """
        Extrait un échantillon aléatoire de gaps
        Utile pour validation ou tests statistiques
        
        Args:
            n_samples: Nombre d'échantillons
            output_file: Fichier de sortie (None = retour array)
        """
        print(f"\n📦 Extraction de {n_samples:,} gaps aléatoires...")
        
        total_gaps = self.metadata['total_gaps']
        
        # Positions aléatoires à échantillonner
        sample_positions = np.sort(np.random.choice(total_gaps, n_samples, replace=False))
        
        samples = []
        current_pos = 0
        sample_idx = 0
        
        for chunk in self.decode_gaps_streaming():
            for gap in chunk:
                if sample_idx < len(sample_positions) and current_pos == sample_positions[sample_idx]:
                    samples.append(gap)
                    sample_idx += 1
                
                current_pos += 1
                
                if sample_idx >= len(sample_positions):
                    break
            
            if sample_idx >= len(sample_positions):
                break
        
        samples = np.array(samples, dtype=np.uint16)
        
        if output_file:
            np.save(output_file, samples)
            print(f"✓ Échantillon sauvegardé: {output_file}")
        
        print(f"✓ {len(samples):,} gaps échantillonnés")
        print(f"  Min: {samples.min()}, Max: {samples.max()}, Mean: {samples.mean():.2f}")
        
        return samples


def main():
    parser = argparse.ArgumentParser(
        description="Analyse des gaps entre nombres premiers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:

  # Statistiques de base
  python analyze_gaps_streaming.py gaps_data/gaps_to_1e13.dat --stats
  
  # Calcul indice de persistance pour p=2
  python analyze_gaps_streaming.py gaps_data/gaps_to_1e13.dat --persistence 2
  
  # Graphique de distribution
  python analyze_gaps_streaming.py gaps_data/gaps_to_1e13.dat --plot
  
  # Échantillonnage
  python analyze_gaps_streaming.py gaps_data/gaps_to_1e13.dat --sample 10000 --output sample.npy
        """
    )
    
    parser.add_argument('gaps_file', help='Fichier de gaps à analyser')
    parser.add_argument('--metadata', help='Fichier de métadonnées (auto si non spécifié)')
    parser.add_argument('--stats', action='store_true', help='Calculer les statistiques')
    parser.add_argument('--persistence', type=int, help='Calculer I(p,N) pour p donné')
    parser.add_argument('--plot', action='store_true', help='Afficher distribution')
    parser.add_argument('--sample', type=int, help='Échantillonner N gaps aléatoires')
    parser.add_argument('--output', help='Fichier de sortie')
    parser.add_argument('--max-gap', type=int, default=100, 
                       help='Gap max pour le plot (défaut: 100)')
    
    args = parser.parse_args()
    
    # Créer l'analyseur
    analyzer = GapsAnalyzer(args.gaps_file, args.metadata)
    
    # Exécuter les analyses demandées
    if args.stats:
        stats = analyzer.compute_statistics()
        
        if args.output:
            import json
            with open(args.output, 'w') as f:
                # Convertir les clés int en str pour JSON
                stats['gap_distribution'] = {str(k): v for k, v in stats['gap_distribution'].items()}
                json.dump(stats, f, indent=2)
            print(f"\n✓ Statistiques sauvegardées: {args.output}")
    
    if args.persistence:
        persistence = analyzer.compute_persistence_index(p=args.persistence)
        
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(persistence, f, indent=2)
            print(f"\n✓ Résultats sauvegardés: {args.output}")
    
    if args.plot:
        analyzer.plot_gap_distribution(
            max_gap=args.max_gap,
            output_file=args.output
        )
    
    if args.sample:
        analyzer.sample_gaps(
            n_samples=args.sample,
            output_file=args.output
        )
    
    # Si aucune option, afficher l'aide
    if not (args.stats or args.persistence or args.plot or args.sample):
        parser.print_help()


if __name__ == "__main__":
    main()
