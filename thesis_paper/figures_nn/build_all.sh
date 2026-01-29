#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build

# Determine tectonic path
if [ -f "$HOME/anaconda3/envs/latex/bin/tectonic" ]; then
    TECTONIC="$HOME/anaconda3/envs/latex/bin/tectonic"
elif command -v tectonic &> /dev/null; then
    TECTONIC="tectonic"
else
    echo "Error: tectonic not found."
    exit 1
fi

echo "Using tectonic: $TECTONIC"

for f in fig_*.tex; do
  echo "==> building $f"
  "$TECTONIC" "$f" --outdir build
done
echo "Done."
ls -lh build/*.pdf
