#!/bin/bash
# Full training pipeline
python pipeline.py \
    --labeled ./train_labeled \
    --original ./train_original \
    --test-labeled ./test_labeled \
    --test-original ./test_original \
    --output ./output \
    --epochs 100 \
    --radius 8 \
    --name polystyrene_model
