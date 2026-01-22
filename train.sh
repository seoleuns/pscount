#!/bin/bash
# Full training pipeline (Training only)
python pipeline.py \
    --labeled ./train_labeled \
    --original ./train_original \
    --output ./output \
    --epochs 100 \
    --radius 8 \
    --name polystyrene_model
