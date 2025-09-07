#!/bin/bash

# Change to the directory containing this script (root of the project)
cd "$(dirname "$0")"

# Train the baselines
cd baselines
datasets=("TVT" "Lo" "Hi" "TVT-KD" "Lo-KD" "Hi-KD")
for dataset in "${datasets[@]}"; do
    python train_dnn.py "$dataset"
    python train_chemprop.py "$dataset"
    python train_bambu.py "$dataset"
done

# Assess the baselines
cd ../assessment
test_datasets=("TVT" "Lo" "Hi")
for dataset in "${test_datasets[@]}"; do
    python assess_baselines.py "$dataset"
    echo "Results saved to:"
    echo "- results/aggregated_baseline_assessment_$dataset.json"
    echo "- results/target_baseline_assessment_$dataset.json"
done

echo "Done."
