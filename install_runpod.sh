#!/bin/bash
#
# Script d'installation automatique - Générateur Gaps Parallèle
# À lancer sur votre Runpod pour configurer l'environnement complet
#
# Usage: bash install_runpod.sh
#

set -e  # Arrêt en cas d'erreur

echo "=================================="
echo "🚀 INSTALLATION GAPS GENERATOR"
echo "=================================="
echo ""

# Couleurs pour les messages
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Fonction pour afficher les messages
info() {
    echo -e "${BLUE}ℹ ${1}${NC}"
}

success() {
    echo -e "${GREEN}✅ ${1}${NC}"
}

error() {
    echo -e "${RED}❌ ${1}${NC}"
}

# 1. Vérifier l'environnement
info "Vérification de l'environnement..."

# Vérifier Python
if ! command -v python3.13 &> /dev/null; then
    error "Python 3.13 non trouvé"
    exit 1
fi
success "Python 3.13 détecté"

# Vérifier pip
if ! command -v pip &> /dev/null; then
    error "pip non trouvé"
    exit 1
fi
success "pip détecté"

# 2. Créer la structure de répertoires
info "Création de la structure de répertoires..."

cd /workspace
mkdir -p NPGaps/{data,results,logs,scripts}
cd NPGaps

success "Répertoires créés: /workspace/NPGaps"

# 3. Installation des dépendances Python
info "Installation des dépendances Python..."

pip install --quiet --upgrade pip
pip install --quiet numpy scipy matplotlib primesieve psutil tqdm

# Vérifier les installations
python3.13 -c "import numpy; import scipy; import matplotlib; import primesieve; import psutil; import tqdm" 2>/dev/null

if [ $? -eq 0 ]; then
    success "Toutes les dépendances installées"
else
    error "Erreur lors de l'installation des dépendances"
    exit 1
fi

# 4. Afficher les informations système
info "Configuration système détectée:"
echo ""

# CPU
CPU_COUNT=$(python3.13 -c "import multiprocessing; print(multiprocessing.cpu_count())")
echo "  • CPU cores: $CPU_COUNT"

# RAM
RAM_GB=$(python3.13 -c "import psutil; print(f'{psutil.virtual_memory().total / 1024**3:.1f}')")
echo "  • RAM totale: ${RAM_GB} GB"

# Python version
PYTHON_VERSION=$(python3.13 --version)
echo "  • Python: $PYTHON_VERSION"

# Primesieve version
PRIMESIEVE_VERSION=$(python3.13 -c "import primesieve; print(primesieve.libprimesieve_version())")
echo "  • Primesieve: $PRIMESIEVE_VERSION"

echo ""

# 5. Créer un script de test rapide
info "Création du script de test..."

cat > test_environment.py << 'EOF'
#!/usr/bin/env python3.13
"""Test rapide de l'environnement"""

import numpy as np
import primesieve
import psutil
import multiprocessing as mp

def test_primesieve():
    """Test primesieve jusqu'à 10^6"""
    primes = primesieve.primes(1000000)
    assert len(primes) == 78498
    return True

def test_numpy():
    """Test NumPy"""
    arr = np.random.rand(1000)
    assert arr.shape == (1000,)
    return True

def test_multiprocessing():
    """Test multiprocessing"""
    cpu_count = mp.cpu_count()
    assert cpu_count > 0
    return True

if __name__ == "__main__":
    print("🧪 Tests de validation:")
    print()
    
    try:
        test_primesieve()
        print("✅ Primesieve: OK")
    except Exception as e:
        print(f"❌ Primesieve: {e}")
    
    try:
        test_numpy()
        print("✅ NumPy: OK")
    except Exception as e:
        print(f"❌ NumPy: {e}")
    
    try:
        test_multiprocessing()
        print("✅ Multiprocessing: OK")
    except Exception as e:
        print(f"❌ Multiprocessing: {e}")
    
    print()
    print(f"💻 CPU: {mp.cpu_count()} cores")
    print(f"💾 RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB")
    print()
    print("✅ Environnement prêt!")
EOF

chmod +x test_environment.py

# 6. Lancer le test
info "Lancement des tests de validation..."
echo ""

python3.13 test_environment.py

if [ $? -eq 0 ]; then
    echo ""
    success "Tests passés avec succès !"
else
    error "Échec des tests"
    exit 1
fi

# 7. Créer des scripts utilitaires
info "Création des scripts utilitaires..."

# Script de monitoring
cat > monitor.sh << 'EOF'
#!/bin/bash
# Monitoring en temps réel

echo "=== Monitoring des Ressources ==="
echo ""
watch -n 5 "
echo '=== CPU ==='
mpstat 1 1 | tail -1
echo ''
echo '=== RAM ==='
free -h | grep 'Mem:'
echo ''
echo '=== Disque ==='
df -h /workspace | tail -1
echo ''
echo '=== Processus Python ==='
ps aux | grep python | grep -v grep | head -3
"
EOF

chmod +x monitor.sh

# Script de nettoyage
cat > cleanup.sh << 'EOF'
#!/bin/bash
# Nettoyage des fichiers temporaires

echo "Nettoyage des fichiers temporaires..."
rm -rf /tmp/*.txt
rm -rf /workspace/NPGaps/__pycache__
echo "✅ Nettoyage terminé"
EOF

chmod +x cleanup.sh

success "Scripts utilitaires créés"

# 8. Créer un fichier de configuration par défaut
info "Création de la configuration par défaut..."

cat > config.json << EOF
{
  "default_workers": $(($CPU_COUNT - 2)),
  "default_buffer_gb": 32,
  "output_dir": "data",
  "checkpoint_interval": 500,
  "monitoring_interval": 10
}
EOF

success "Configuration créée: config.json"

# 9. Instructions finales
echo ""
echo "=================================="
echo "✅ INSTALLATION TERMINÉE"
echo "=================================="
echo ""
echo "📁 Répertoire: /workspace/NPGaps"
echo ""
echo "🎯 Prochaines étapes:"
echo ""
echo "1. Uploader vos scripts:"
echo "   • generate_gaps_parallel.py"
echo "   • benchmark_parallel.py"
echo ""
echo "2. Test rapide (10^9):"
echo "   python3.13 generate_gaps_parallel.py --target 1e9 --workers 8"
echo ""
echo "3. Benchmark:"
echo "   python3.13 benchmark_parallel.py quick"
echo ""
echo "4. Génération 10^13:"
echo "   python3.13 generate_gaps_parallel.py --target 1e13 --workers 28 --buffer 64"
echo ""
echo "💡 Scripts utilitaires:"
echo "   • ./test_environment.py  → Tester l'environnement"
echo "   • ./monitor.sh           → Surveiller les ressources"
echo "   • ./cleanup.sh           → Nettoyer les temporaires"
echo ""
echo "📚 Documentation:"
echo "   • README_QUICKSTART.md"
echo "   • GUIDE_UTILISATION_PARALLELE.md"
echo ""
echo "🚀 Bon calcul !"
echo ""
