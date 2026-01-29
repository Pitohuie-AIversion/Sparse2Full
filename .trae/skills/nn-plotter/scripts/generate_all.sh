#!/bin/bash
# Wrapper to run the project's figure generation
set -e

# Get project root (assuming this script is in .trae/skills/nn-plotter/scripts/)
# We want to run from the project root.
# If we are already at project root (which is typical for the agent), we can just check for the target dir.

if [ -d "thesis_paper/figures_nn" ]; then
    PROJECT_ROOT="."
elif [ -d "../../../thesis_paper/figures_nn" ]; then
    # In case we are running from the script dir
    PROJECT_ROOT="../../.."
else
    # Try to find it
    if [ -f "/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/thesis_paper/figures_nn/export_and_gen_tikz.py" ]; then
        PROJECT_ROOT="/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full"
    else
        echo "Error: Cannot find project root with 'thesis_paper/figures_nn'"
        exit 1
    fi
fi

cd "$PROJECT_ROOT"

echo "Generating TikZ for all models..."
# Add all known models here
MODELS="edsr,unet,fno2d,segformer,swint,swin_unet,hybrid,liif,mlp_mixer,deeponet,ufno"

python thesis_paper/figures_nn/export_and_gen_tikz.py --models "$MODELS"

echo "Compiling PDFs..."
bash thesis_paper/figures_nn/build_all.sh
