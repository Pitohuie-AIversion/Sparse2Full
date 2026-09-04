#!/bin/bash
# Wrapper to run the project's figure generation
set -e

# Get project root (this script is in .trae/skills/nn-plotter/scripts/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

if [ ! -f "$PROJECT_ROOT/thesis_paper/figures_nn/export_and_gen_tikz.py" ]; then
    echo "Error: Cannot find project root with 'thesis_paper/figures_nn'"
    exit 1
fi

cd "$PROJECT_ROOT"

echo "Generating TikZ for all models..."
# Add all known models here
MODELS="edsr,unet,fno2d,segformer,swint,swin_unet,hybrid,liif,mlp_mixer,deeponet,ufno"

python thesis_paper/figures_nn/export_and_gen_tikz.py --models "$MODELS"

echo "Compiling PDFs..."
bash thesis_paper/figures_nn/build_all.sh
