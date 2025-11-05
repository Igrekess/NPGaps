#!/bin/bash

set -e  # Arrête le script en cas d'erreur

echo "=========================================="
echo "Installation de NPGaps sur RunPod"
echo "=========================================="

# 1. Mise à jour du système et installation des dépendances de base
echo "📦 Installation des dépendances système..."
apt update
apt install -y \
    cmake \
    g++ \
    git \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    wget

# 2. Installation de primesieve
echo "🔧 Compilation et installation de primesieve..."
cd /tmp
rm -rf primesieve  # Nettoyage si existe déjà
git clone https://github.com/kimwalisch/primesieve.git
cd primesieve
cmake -DCMAKE_INSTALL_PREFIX=/usr/local .
make -j$(nproc)
make install
ldconfig

# Mise à jour du PATH pour inclure /usr/local/bin
export PATH="/usr/local/bin:$PATH"
echo 'export PATH="/usr/local/bin:$PATH"' >> ~/.bashrc

# Vérification de l'installation de primesieve
echo "✓ Vérification de primesieve..."
/usr/local/bin/primesieve --version 2>/dev/null || primesieve --version || echo "⚠️ Erreur: primesieve non trouvé"

# 3. Clone de NPGaps
echo "📥 Clonage de NPGaps..."
cd /workspace  # Répertoire standard RunPod
rm -rf NPGaps  # Nettoyage si existe déjà
git clone https://github.com/igrekess/NPGaps.git
cd NPGaps

# 4. Installation des dépendances Python
echo "🐍 Installation des dépendances Python..."
pip3 install --upgrade pip

# Installation des packages Python courants pour l'analyse de nombres premiers
pip3 install \
    numpy \
    scipy \
    matplotlib \
    pandas \
    tqdm \
    psutil

# 5. Vérification de l'installation
echo ""
echo "=========================================="
echo "✅ Installation terminée!"
echo "=========================================="
echo ""
echo "Vérifications:"
python3 --version
pip3 --version
echo ""
python3 -c "import primesieve; print(f'primesieve Python: {primesieve.__version__}')" || echo "⚠️ Module primesieve Python non trouvé"
echo ""
echo "📁 Répertoire NPGaps: /workspace/NPGaps"
echo ""
echo "Pour lancer JupyterLab:"
echo "  cd /workspace/NPGaps && jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root"