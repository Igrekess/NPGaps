# Génération de Gaps entre Nombres Premiers par Streaming

Scripts pour générer et analyser les gaps entre nombres premiers jusqu'à 10^15, avec **RAM constante** (2-4 GB).

## 🎯 Caractéristiques

- ✅ **RAM constante** : 2-4 GB quelle que soit la cible (10^9, 10^13, 10^15)
- ✅ **Checkpoints automatiques** : Reprise possible après interruption
- ✅ **Validation d'intégrité** : Hash SHA-256 de chaque segment
- ✅ **Streaming complet** : Génération ET analyse sans charger tout en RAM
- ✅ **Compatible théorie de la persistance** : Calcul de I(p,N) intégré

## 📦 Installation

```bash
# Installer primesieve (requis)
pip install primesieve numpy matplotlib

# Télécharger les scripts
# generate_gaps_streaming.py
# analyze_gaps_streaming.py
```

## 🚀 Utilisation

### 1. Génération des gaps

#### Exemple 1 : Jusqu'à 10^11 (rapide, ~2 min)
```bash
python generate_gaps_streaming.py --target 1e11
```

**Résultat :**
- Fichier : `gaps_data/gaps_to_1e+11.dat` (3.9 GB)
- Métadonnées : `gaps_data/metadata_1e+11.json`
- RAM utilisée : ~2 GB
- Temps : ~2 minutes

#### Exemple 2 : Jusqu'à 10^13 (chez vous, ~2h)
```bash
python generate_gaps_streaming.py --target 1e13 --output my_gaps
```

**Résultat :**
- Fichier : `my_gaps/gaps_to_1e+13.dat` (334 GB)
- RAM utilisée : ~2-4 GB
- Temps : ~2 heures
- Coût : Gratuit (chez vous)

#### Exemple 3 : Jusqu'à 10^15 (RunPod, ~10h)
```bash
python generate_gaps_streaming.py --target 1e15 --segment-size 1e11
```

**Résultat :**
- Fichier : `gaps_data/gaps_to_1e+15.dat` (~29 TB)
- RAM utilisée : ~4 GB
- Temps : ~10 heures
- Coût : $3-5 sur RunPod

### 2. Reprise après interruption

Si la génération est interrompée (Ctrl+C, crash, etc.), elle reprendra automatiquement :

```bash
# Relancez simplement la même commande
python generate_gaps_streaming.py --target 1e13
# → Détecte le checkpoint et reprend où ça s'était arrêté
```

### 3. Analyse des gaps

#### Statistiques de base
```bash
python analyze_gaps_streaming.py gaps_data/gaps_to_1e13.dat --stats
```

**Affiche :**
- Gap minimum, maximum, moyen
- Écart-type
- Top 10 gaps les plus fréquents
- Distribution complète

#### Calcul indice de persistance I(p,N)
```bash
# Pour p=2 (projection Z/(4)Z)
python analyze_gaps_streaming.py gaps_data/gaps_to_1e13.dat --persistence 2

# Pour p=3 (projection Z/(6)Z)
python analyze_gaps_streaming.py gaps_data/gaps_to_1e13.dat --persistence 3

# Pour p=5
python analyze_gaps_streaming.py gaps_data/gaps_to_1e13.dat --persistence 5
```

**Calcule :**
- Entropie de Shannon
- Information mutuelle
- Taux de concentration
- Distribution dans l'espace modulaire

#### Visualisation
```bash
# Histogramme des gaps jusqu'à 100
python analyze_gaps_streaming.py gaps_data/gaps_to_1e13.dat --plot --max-gap 100 --output distribution.png
```

#### Échantillonnage
```bash
# Extraire 100k gaps aléatoires pour tests statistiques
python analyze_gaps_streaming.py gaps_data/gaps_to_1e13.dat --sample 100000 --output sample.npy

# Puis utiliser l'échantillon
import numpy as np
gaps = np.load('sample.npy')
```

### 4. Vérification d'intégrité

```bash
python generate_gaps_streaming.py --verify gaps_data/gaps_to_1e13.dat
```

Vérifie :
- Taille du fichier
- Nombre de gaps
- Checksums (si disponibles)

## 📊 Capacités par configuration

| RAM locale | Cible max recommandée | Temps | Stockage |
|------------|----------------------|-------|----------|
| 8 GB | 10^11 | 2 min | 4 GB |
| 16 GB | 10^13 | 2 h | 334 GB |
| 32 GB | 10^13 | 2 h | 334 GB |
| RunPod 256 GB | 10^15 | 10 h | 29 TB |

**Note :** La RAM est constante grâce au streaming, donc même 8 GB peut théoriquement aller jusqu'à 10^15 (mais le temps sera très long).

## 📁 Structure des fichiers

### Fichier de gaps (`.dat`)
```
Format binaire : suite de bytes uint8
- Si gap < 255 : valeur directe
- Si gap >= 255 : [255, high_byte, low_byte]

Exemple : [2, 4, 6, 2, 255, 1, 0, 4, ...]
          └─ 2, 4, 6, 2, 256, 4...
```

### Fichier de métadonnées (`.json`)
```json
{
  "target": 1e13,
  "first_prime": 2,
  "last_prime": 9999999999971,
  "total_gaps": 346065536839,
  "total_bytes": 346123456789,
  "segments_processed": 1000,
  "start_time": "2025-11-05T10:00:00",
  "end_time": "2025-11-05T12:00:00",
  "checksums": {
    "segment_1": "a1b2c3...",
    "segment_2": "d4e5f6...",
    ...
  }
}
```

## 🔬 Intégration avec la Théorie de la Persistance

### Calcul de I(p,N) pour plusieurs p

```python
from analyze_gaps_streaming import GapsAnalyzer

analyzer = GapsAnalyzer('gaps_data/gaps_to_1e13.dat')

# Calculer I(p,N) pour les premiers p
primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
results = []

for p in primes:
    I_p = analyzer.compute_persistence_index(p)
    results.append(I_p)
    print(f"I({p}, 10^13) = {I_p['I_p']:.6f} bits")

# Fitter le modèle exponentiel
# I(p,N) = I_max(N) * [1 - exp(-k(N) * p)]
```

### Analyse multi-échelle

```bash
# Générer plusieurs décades
python generate_gaps_streaming.py --target 1e10
python generate_gaps_streaming.py --target 1e11
python generate_gaps_streaming.py --target 1e12
python generate_gaps_streaming.py --target 1e13

# Analyser chaque décade
for target in 1e10 1e11 1e12 1e13; do
    python analyze_gaps_streaming.py gaps_data/gaps_to_${target}.dat \
        --persistence 2 --output results_${target}.json
done

# Extraire les k_eff(N) et I_max(N) pour chaque échelle N
```

## ⚡ Optimisations

### 1. Taille de segment optimale

```bash
# Pour 10^11 : segments de 10^9 (rapide)
python generate_gaps_streaming.py --target 1e11 --segment-size 1e9

# Pour 10^13 : segments de 10^10 (optimal)
python generate_gaps_streaming.py --target 1e13 --segment-size 1e10

# Pour 10^15 : segments de 10^11 (RunPod)
python generate_gaps_streaming.py --target 1e15 --segment-size 1e11
```

### 2. Disque SSD recommandé

Pour les grandes générations (10^13+), utilisez un **SSD NVMe** :
- Vitesse d'écriture : 2-7 GB/s
- Évite le bottleneck disque
- Réduit le temps total de 20-30%

### 3. Parallélisation (future)

Pour l'instant, la génération est séquentielle. Une version parallèle pourrait :
- Diviser en segments indépendants
- Générer en parallèle sur plusieurs cœurs
- Fusionner les résultats

## 🐛 Debugging

### Problème : Génération très lente
```bash
# Vérifier la vitesse du disque
dd if=/dev/zero of=test.dat bs=1M count=1024
rm test.dat

# Si < 100 MB/s → Utiliser un SSD
```

### Problème : Out of Memory
```bash
# Réduire la taille des segments
python generate_gaps_streaming.py --target 1e13 --segment-size 1e9

# La RAM ne devrait jamais dépasser 4-5 GB
```

### Problème : Fichier corrompu
```bash
# Vérifier l'intégrité
python generate_gaps_streaming.py --verify gaps_data/gaps_to_1e13.dat

# Si corrompu, supprimer et régénérer
rm gaps_data/gaps_to_1e13.dat
rm gaps_data/metadata_1e+13.json
rm gaps_data/checkpoint_1e+13.json
```

## 📈 Roadmap

### Implémenté ✅
- [x] Génération streaming avec RAM constante
- [x] Checkpoints et reprise automatique
- [x] Analyse streaming (stats, I(p,N))
- [x] Validation d'intégrité
- [x] Visualisation

### À venir 🚧
- [ ] Parallélisation multi-cœurs
- [ ] Compression à la volée (gzip/zstd)
- [ ] Interface web pour monitoring
- [ ] Export vers formats standards (HDF5, Parquet)
- [ ] Calcul distribué (multi-machines)

## 📞 Support

Pour toute question ou problème :
1. Vérifier que primesieve est bien installé : `pip show primesieve`
2. Tester avec une petite cible d'abord : `--target 1e9`
3. Vérifier l'espace disque disponible : `df -h`

## 📜 Licence

Scripts pour le projet Théorie de la Persistance.
Libre d'utilisation pour la recherche académique.

## 🙏 Remerciements

- Kim Walisch pour primesieve
- Communauté des chercheurs en théorie des nombres
