#!/bin/bash
# Evaluate: Cellpose default vs PSCount (fine-tuned)

## 1. Ground Truth by red-pointed images
# Cellpose default (no fine-tuning)
python pipeline.py \
    --model default \
    --test-labeled ./test_labeled \
    --test-original ./test_original \
    --output ./results_default

# PSCount (fine-tuned) 
python pipeline.py \
    --model ./output/models/polystyrene_model \
    --test-labeled ./test_labeled \
    --test-original ./test_original \
    --output ./results_pscount

## 2. Ground Truth by CSV
# Cellpose default (no fine-tuning)
python pipeline.py \
    --model default \
    --input ./test_original_csv \
    --csv ./test_ground_truth.csv \
    --output ./results_default_csv

# PSCount (fine-tuned) 
python pipeline.py \
    --model ./output/models/polystyrene_model \
    --input ./test_original_csv \
    --csv ./test_ground_truth.csv \
    --output ./results_pscount_csv
