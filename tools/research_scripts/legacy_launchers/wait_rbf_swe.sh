#!/bin/bash
while true; do
  if [ -f "runs_rbf_cnn_swe/RBFCNN-SWE-model_RBFCNN-s2025-20260515/test_results.json" ] || [ -f "runs_rbf_cnn_swe/test_results.json" ]; then
    break
  fi
  sleep 10
done
echo "RBF-CNN SWE finished"
