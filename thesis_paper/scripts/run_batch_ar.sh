#!/usr/bin/env bash
set -euo pipefail

BASE_CONFIG=${1:-thesis_paper/configs/spatial_training_template.yaml}
MODEL=${2:-UNet}
BATCHES=${3:-64,96,128}
SEEDS=${4:-2025,2026,2027}

LOG_DIR="runs/batch_sweep/${MODEL}"
mkdir -p "${LOG_DIR}"

IFS=',' read -ra BS_ARR <<< "${BATCHES}"
BEST=""

for BS in "${BS_ARR[@]}"; do
  echo "Starting ${MODEL} batch_size=${BS}"
  TMPDIR=$(mktemp -d -t batchcfg_XXXX)
  TMPCFG="${TMPDIR}/cfg_bs${BS}.yaml"
  python - <<PY "${BASE_CONFIG}" "${TMPCFG}" "${BS}"
import sys, yaml
base = sys.argv[1]
out = sys.argv[2]
bs = int(sys.argv[3])
with open(base, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
for key in ('training','data'):
    c = cfg.get(key, {})
    dl = c.get('dataloader', {})
    dl['batch_size'] = bs
    dl['val_batch_size'] = max(1, bs//2)
    dl['test_batch_size'] = max(1, bs//2)
    c['dataloader'] = dl
    cfg[key] = c
with open(out, 'w', encoding='utf-8') as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PY

  LOGFILE="${LOG_DIR}/${MODEL}_bs${BS}.log"
  set +e
  python tools/training/train_real_data_ar.py --config "${TMPCFG}" --model "${MODEL}" --seeds "${SEEDS}" >"${LOGFILE}" 2>&1
  CODE=$?
  set -e
  if [ "${CODE}" -eq 0 ]; then
    BEST="${BS}"
    echo "Completed ${MODEL} batch_size=${BS}"
  else
    echo "failed_batch_size=${BS}"
    echo "see_log=${LOGFILE}"
    break
  fi
done

if [ -z "${BEST}" ]; then
  echo "no batch size succeeded"
  exit 1
fi
echo "best_batch_size=${BEST}"
exit 0
