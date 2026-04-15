
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

# Install other dependencies
pip install -r requirements.txt
```

## CellPose-SAM 4.0
#pretrained model:
https://huggingface.co/mouseland/cellpose-sam/blob/main/cpsam

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

## Data and Model
Full dataset and pre-trained model available at:
https://doi.org/10.5281/zenodo.18364078

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
python pipeline.py \
    --labeled ./train_labeled \
    --original ./train_original \
    --output ./output \
    --epochs 100 \
    --radius 8 \
    --name polystyrene_model
```

Evaluate with ground truth:

```bash
python pipeline.py \
    --model ./output/models/polystyrene_model \
    --test-labeled ./test_labeled \
    --test-original ./test_original \
    --output ./results_pscount
```

#### 4. Predict

Count particles in new images without ground truth or labeled images:

```bash
python predict.py \
    --model ./output/models/polystyrene_model \
    --input ./new_images \
    --output ./results_unknown
```

## Shell Scripts
Shell scripts (`train.sh`, `evaluate.sh`, `allprocess.sh`) are provided for convenience.


## Parameters

| Parameter  |Default | Description               |
|----------- |--------|---------------------------|
| `--radius` | 8      | Mask radius in pixels     |
| `--epochs` | 100    | Number of training epochs |
| `--lr`     | 1e-5   | Learning rate             |

Note: Weight decay is set to 0.1 (Cellpose-SAM default) and is not exposed as a command-line argument.

## Estimating Mask Radius for Your Imaging Setup

To determine the appropriate mask radius for your microscopy images, 
use the provided utility script:

```bash
python estimate_mask_radius.py --image your_image.tif --visualize
```

The script measures the apparent diameter of isolated single particles 
and recommends an appropriate mask radius. Adjust `--min_area` and 
`--max_area` based on your particle size if needed.

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
# OpenCV (with labeled ground truth image)
python opencv_counting.py --input ./test_original --labeled ./test_labeled --output ./results_opencv

# OpenCV (with csv for ground truth)
python opencv_counting.py --input ./test_original_csv --csv ./test_ground_truth.csv --output ./results_opencv_csv
```
## Performance

| Dataset     | Training Images | Test Images | MAE |
|-------------|-----------------|-------------|-----|
| Polystyrene | 19              | 24          | 2.1 |

Baseline Cellpose (without fine-tuning): MAE = 65.5

All experiments were performed on a server equipped with an NVIDIA A6000 GPU (48GB VRAM). Training on 19 images took approximately 4.7 minutes, and inference time was 1.97 ± 0.15 seconds per image (1500 × 1500 pixels, n=24 in Test set A and B). 

## Reproducibility
Training results may vary slightly between runs due to
stochastic elements in deep learning. For reproducible
results, use the provided pre-trained model (https://doi.org/10.5281/zenodo.18364078).

## Citation

If you use this tool in your research, please cite:

```bibtex
@article{shin2026polystyrene,
  title={Polystyrene Particle Counter: A Deep Learning Pipeline for Automated Particle Counting},
  author={Shin, Seoleun and Lee, Ji Youn},
  journal={SoftwareX (submitted)},
  year={2026}
}
```

## License

MIT License

## Acknowledgments

- [Cellpose](https://github.com/MouseLand/cellpose) - Base segmentation framework
- [Cellpose-SAM](https://www.biorxiv.org/content/10.1101/2025.04.28.651001) - Base foundation model (Pachitariu et al., 2025)
