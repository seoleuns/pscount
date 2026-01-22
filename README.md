
# Polystyrene Particle Counter (PSCount)

A deep learning pipeline for automated counting of polystyrene particles in microscopy images using fine-tuned Cellpose models.

## Overview

This tool enables accurate particle counting with minimal manual annotation. Instead of creating detailed segmentation masks, users only need to mark particle centers with red points. The pipeline automatically converts these point annotations to training masks and fine-tunes a Cellpose model for your specific data.

## Features

- **Lightweight annotation**: Mark particles with red points instead of detailed masks
- **Automatic mask generation**: Convert point annotations to training masks
- **Transfer learning**: Fine-tuned Cellpose models with small datasets
- **High accuracy**: Achieve MAE < 5 with only 19 training images
- **Visualization**: Generate annotated output images with particle counts

## Installation

### Requirements

- Python 3.10 (recommended)
- CUDA-compatible GPU (recommended)

### Setup

```bash
# Create conda environment
conda create -n pscount python=3.10
conda activate pscount

# Install PyTorch with CUDA support
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia

# Install dependencies
pip install cellpose
pip install tifffile pandas opencv-python matplotlib
```
## CellPose-SAM 4.0
#pretrained model:
https://huggingface.co/mouseland/cellpose-sam/blob/main/cpsam

## Usage

### Quick Start (End-to-End Pipeline)

With test data:

```bash
python pipeline.py \
    --labeled ./train_labeled \
    --original ./train_original \
    --epochs 100 \
    --test-labeled ./test_labeled \
    --test-original ./test_original \
    --output ./output
```

## Usage Modes (with shell scripts or command lines)

### Mode 1: Full Training Pipeline
```bash
bash scripts/train.sh
# or
python pipeline.py --labeled ./train_labeled --original ./train_original --epochs 100
```

### Mode 2: Evaluation I (Cellpose default vs. PSCount When Ground Truth Dataset is available: red pointed images or a csv file)
```bash
bash scripts/evaluate.sh

# Cellpose default
python pipeline.py --model default --test-labeled ./test_labeled --test-original ./test_original --output ./results_default

# Fine-tuned (PSCount)
python pipeline.py --model ./output/models/polystyrene_model --test-labeled ./test_labeled --test-original ./test_original --output./results_test

# or

python pipeline.py --model ./output/models/polystyrene_model --input ./test_csv_original --csv ./test_ground_truth.csv --output ./results_test_csv
```

### Mode 4: Prediction Only
```bash
bash scripts/predict.sh
# or
python pipeline.py --model ./output/models/polystyrene_model --input ./new_images --output ./results
```

### Step-by-Step Usage from Data Pretreatment and

#### 1. Extract Point Coordinates

Extract red point coordinates from labeled images:

```bash
python extract_points.py \
    --input ./train_labeled \
    --output coordinates.csv
```

#### 2. Create Training Masks

Generate Cellpose-compatible masks from coordinates:

```bash
python create_masks.py \
    --csv coordinates.csv \
    --original ./train_original \
    --output ./train_data \
    --radius 8
```

#### 3. Train Model

Fine-tune Cellpose on your data:

```bash
python train.py \
    --train ./train_data \
    --test ./test_data \
    --name polystyrene_model \
    --epochs 100
```

Evaluate with ground truth:

```bash
python predict.py \
    --model ./models/polystyrene_model \
    --input ./test_original \
    --labeled ./test_labeled \
    --output ./results_test
```

#### 4. Predict

Count particles in new images without ground truth or labeled images:

```bash
python predict.py \
    --model ./models/polystyrene_model \
    --input ./new_images \
    --output ./results_unknown
```

## Data Preparation

### File Structure

```
project/
├── original_images/          # Original microscopy images
│   ├── sample_001.tif
│   ├── sample_002.tif
│   └── ...
├── labeled_images/           # Images with red point annotations
│   ├── sample_001_150count_Flatten.tif
│   ├── sample_002_143count_Flatten.tif
│   └── ...
└── polystyrene_counter/      # This repository
```

### Labeling Guidelines

1. Open original image in ImageJ/FIJI or similar software
2. Mark each particle center with a red point 
3. Save as TIF with naming convention: `{original_name}_{count}count_Flatten.tif`

### Labeling Tips

- Use consistent red color (R > 150, G < 100, B < 100)
- Place points at particle centers
- Include particle count in filename for validation

## Parameters

| Parameter  |Default | Description               |
|----------- |--------|---------------------------|
| `--radius` | 8      | Mask radius in pixels     |
| `--epochs` | 100    | Number of training epochs |
| `--lr`     | 1e-5   | Learning rate             |

## Output

```
output/
├── coordinates.csv           # Extracted point coordinates
├── train_data/               # Training images and masks
│   ├── sample_001.tif
│   ├── sample_001_mask_check.png
│   ├── sample_001_masks.npy
│   └── ...
├── models/                   # Trained model
│   └── polystyrene_model
|
├── sample_001_result.png
├── sample_001_prob_map.png
├── evaluation_results.csv
└── ...
```

### OpenCV test (optional for comparions)
```bash
bash scripts/opencv_evaluate.sh
# OpenCV (with labeled ground turth image)
python opencv_counting.py --input ./test_original --labeled ./test_labeled --output ./results_opencv

# OpenCV (with csv for ground turth)
python opencv_counting.py --input ./test_original_csv --csv ./test_ground_truth.csv --output ./results_opencv_csv
```
## Performance

| Dataset     | Training Images | Test Images | MAE |
|-------------|-----------------|-------------|-----|
| Polystyrene | 19              | 24          | 2.2 |

Baseline Cellpose (without fine-tuning): MAE = 65.5

## Reproducibility
Training results may vary slightly between runs due to
stochastic elements in deep learning. For reproducible
results, use the provided pre-trained model.

## Citation

If you use this tool in your research, please cite:

```bibtex
@article{shin2025polystyrene,
  title={Polystyrene Particle Counter: A Deep Learning Pipeline for Automated Particle Counting},
  author={...},
  journal={SoftwareX},
  year={2025}
}
```

## License

MIT License

## Acknowledgments

- [Cellpose](https://github.com/MouseLand/cellpose) - Base segmentation model
- [Cellpose 2.0 paper](https://www.nature.com/articles/s41592-022-01663-4) - Fine-tuning methodology
