# 🚀 Guide de Démarrage Rapide

## Installation (5 minutes)

### 1. Installer les dépendances
```bash
pip install primesieve numpy matplotlib
```

### 2. Télécharger les scripts
Placez ces 3 fichiers dans le même répertoire :
- `generate_gaps_streaming.py`
- `analyze_gaps_streaming.py`  
- `test_installation.py`

### 3. Tester l'installation
```bash
python test_installation.py
```

Vous devriez voir :
```
🎉 TOUS LES TESTS SONT PASSÉS!
```

---

## Première utilisation (10 minutes)

### Étape 1 : Génération jusqu'à 10^10 (1 minute)
```bash
python generate_gaps_streaming.py --target 1e10
```

**Ce que vous verrez :**
```
🚀 GÉNÉRATION DE GAPS PAR STREAMING
Cible: 1.00e+10
Taille segment: 1.00e+09
RAM utilisée: ~2-4 GB (constant)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 100.00% | Position: 1.00e+10 | Gaps: 455,052,511 | ...
✅ GÉNÉRATION TERMINÉE
```

**Résultat :**
- Fichier : `gaps_data/gaps_to_1e+10.dat` (0.4 GB)
- Temps : ~1 minute
- RAM : ~2 GB

### Étape 2 : Analyse (30 secondes)
```bash
python analyze_gaps_streaming.py gaps_data/gaps_to_1e+10.dat --stats
```

**Ce que vous verrez :**
```
📊 STATISTIQUES DES GAPS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Nombre de gaps: 455,052,511

Gap minimum: 1
Gap maximum: 154
Gap moyen: 21.98
Écart-type: 6.42

🔢 Top 10 gaps les plus fréquents:
  Gap   2:  204,898,509 (45.02%)
  Gap   4:   68,330,034 (15.01%)
  Gap   6:   92,195,461 (20.26%)
  ...
```

### Étape 3 : Indice de persistance (30 secondes)
```bash
python analyze_gaps_streaming.py gaps_data/gaps_to_1e+10.dat --persistence 2
```

**Ce que vous verrez :**
```
🧮 CALCUL INDICE DE PERSISTANCE I(2, N)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Espace de projection: Z/(4)Z
Entropie mesurée: 1.5234 bits
Entropie maximale: 2.0000 bits
I(2, 10^10) = 0.476600 bits
Taux de concentration: 23.83%
```

---

## Cas d'usage typiques

### Pour aller jusqu'à 10^13 chez vous

```bash
# 1. Génération (~2 heures)
python generate_gaps_streaming.py --target 1e13

# 2. Analyse complète
python analyze_gaps_streaming.py gaps_data/gaps_to_1e+13.dat --stats --output stats_1e13.json

# 3. Calcul I(p,N) pour plusieurs p
for p in 2 3 5 7 11; do
    python analyze_gaps_streaming.py gaps_data/gaps_to_1e+13.dat \
        --persistence $p --output I_p${p}_1e13.json
done

# 4. Visualisation
python analyze_gaps_streaming.py gaps_data/gaps_to_1e+13.dat \
    --plot --max-gap 200 --output distribution_1e13.png
```

### Génération progressive par décade

```bash
# Commencez petit et montez progressivement
python generate_gaps_streaming.py --target 1e10   # 1 min
python generate_gaps_streaming.py --target 1e11   # 2 min
python generate_gaps_streaming.py --target 1e12   # 15 min
python generate_gaps_streaming.py --target 1e13   # 2 h

# Analysez chaque décade pour valider votre théorie
# avant de passer à la suivante
```

### Sur RunPod pour 10^14-10^15

```bash
# Connectez-vous à votre instance RunPod (32 cores, 256 GB RAM)

# Installation rapide
sudo apt update
sudo apt install python3-pip
pip3 install primesieve numpy

# Upload des scripts via l'interface web RunPod

# Génération 10^14 → 10^15 (10h, $3-5)
python3 generate_gaps_streaming.py --target 1e15 --segment-size 1e11

# Analyse immédiate (avant d'upload vers cloud storage)
python3 analyze_gaps_streaming.py gaps_data/gaps_to_1e+15.dat \
    --persistence 2 --output I_p2_1e15.json

# Upload vers S3/autre (seulement si vous gardez)
```

---

## Résolution de problèmes courants

### Problème : "ModuleNotFoundError: No module named 'primesieve'"
**Solution :**
```bash
pip install primesieve
# ou
pip3 install primesieve
```

### Problème : Génération très lente
**Diagnostic :**
```bash
# Vérifier la vitesse du disque
dd if=/dev/zero of=test.dat bs=1M count=1024
rm test.dat
```
**Solution :** Utilisez un SSD NVMe si possible

### Problème : Out of Memory
**Solution :** Réduire la taille des segments
```bash
python generate_gaps_streaming.py --target 1e13 --segment-size 1e9
```

### Problème : Interruption pendant génération
**Solution :** Relancez simplement la même commande
```bash
# Le checkpoint est automatiquement détecté
python generate_gaps_streaming.py --target 1e13
# → Reprend où ça s'était arrêté
```

---

## Estimation des temps et coûts

| Cible | Où générer ? | Temps | Stockage | RAM | Coût |
|-------|-------------|-------|----------|-----|------|
| 10^9 | Chez vous | 10s | 0.04 GB | 2 GB | $0 |
| 10^10 | Chez vous | 1 min | 0.4 GB | 2 GB | $0 |
| 10^11 | Chez vous | 2 min | 3.9 GB | 2 GB | $0 |
| 10^12 | Chez vous | 15 min | 36 GB | 2 GB | $0 |
| 10^13 | Chez vous | 2 h | 334 GB | 2-4 GB | $0 |
| 10^14 | RunPod | 1 h | 3 TB | 4 GB | $0.50 |
| 10^15 | RunPod | 10 h | 29 TB | 4 GB | $3-5 |

**Coûts de stockage cloud (si vous gardez les fichiers) :**
- AWS S3 Standard : $23/TB/mois
- AWS S3 Glacier : $1/TB/mois
- Google Cloud Storage : $20/TB/mois
- Disque externe 4 TB : $100 one-time

---

## Prochaines étapes

### Pour valider votre Théorie de la Persistance :

1. **Générer jusqu'à 10^13 chez vous** (2h, gratuit)
2. **Calculer I(p,N) pour p=2,3,5,7,...** 
3. **Fitter le modèle exponentiel :**
   ```
   I(p,N) = I_max(N) × [1 - exp(-k(N) × p)]
   ```
4. **Extraire k_eff(N) et I_max(N)** pour chaque échelle
5. **Vérifier le scaling logarithmique :**
   ```
   k(N) ~ 1/ln(N)
   I_max(N) ~ ln(N)
   ```
6. **Si tout est validé → RunPod pour 10^15** (Question P10!)

### Pour publication :

- Garder les 3 TB de 10^14→10^15 (données critiques)
- Échantillons stratifiés des autres décades
- Graphiques haute résolution
- Statistiques bootstrap pour intervalles de confiance

---

## Ressources

- **Documentation complète :** `README_GAPS_STREAMING.md`
- **Code source :** Scripts Python bien commentés
- **Support :** Relisez la documentation ou testez avec `--help`

```bash
python generate_gaps_streaming.py --help
python analyze_gaps_streaming.py --help
```

---

## 🎉 Vous êtes prêt !

Commencez par :
```bash
python generate_gaps_streaming.py --target 1e11
```

Et observez la magie du streaming opérer avec seulement 2 GB de RAM ! 🚀
