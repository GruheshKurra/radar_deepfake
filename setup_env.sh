#!/bin/bash
# Setup script for RADAR project environment

set -e  # Exit on error

echo "🚀 Setting up RADAR Python environment..."

# Source conda
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh

# Check if 'ml' environment exists
if conda env list | grep -q "^ml "; then
    echo "✓ 'ml' environment already exists"
    conda activate ml
else
    echo "📦 Creating 'ml' conda environment with Python 3.11..."
    conda create -n ml python=3.11 -y
    conda activate ml
fi

# Install PyTorch with CUDA support if available
echo "📥 Installing PyTorch..."
if [[ $(uname -m) == 'arm64' ]]; then
    # Mac M1/M2 - use MPS acceleration
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
else
    # Intel Mac or Linux
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
fi

# Install remaining requirements
echo "📥 Installing project dependencies..."
pip install -r requirements.txt

echo "✅ Environment setup complete!"
echo ""
echo "To activate this environment in the future, run:"
echo "  conda activate ml"
echo ""
echo "To deactivate:"
echo "  conda deactivate"
