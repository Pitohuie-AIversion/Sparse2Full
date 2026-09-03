#!/bin/bash
while true; do
  if [ -f "runs_bicubic_cnn_drd/BicubicCNN-Crop16-DRD-model_BicubicCNN-s2025-20260519/test_results.json" ] || [ -f "runs_bicubic_cnn_drd/test_results.json" ]; then
    break
  fi
  sleep 10
done
echo "Bicubic-CNN DRD finished"
