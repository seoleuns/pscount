"""
Fine-tune Cellpose model on custom dataset.
"""
import numpy as np
import tifffile
from pathlib import Path
import argparse
import os

import torch
import numpy as np
import random

def set_seed(seed=2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(2025)



def load_data(folder):
    """
    Load images and masks from a folder.
    
    Args:
        folder: Directory with .tif images and _masks.npy files
    
    Returns:
        Tuple of (images list, labels list)
    """
    images = []
    labels = []
    path = Path(folder)
    
    for tif_file in sorted(path.glob("*.tif")):
        mask_file = path / f"{tif_file.stem}_masks.npy"
        if mask_file.exists():
            images.append(tifffile.imread(str(tif_file)))
            labels.append(np.load(str(mask_file)))
    
    return images, labels


def train_model(train_dir, test_dir=None, model_name="polystyrene_model", 
                save_path="./models", n_epochs=100, learning_rate=1e-5):
    """
    Fine-tune Cellpose model.
    
    Args:
        train_dir: Directory with training data
        test_dir: Directory with test data (optional)
        model_name: Name for the saved model
        save_path: Directory to save the model
        n_epochs: Number of training epochs
        learning_rate: Learning rate
    
    Returns:
        Path to the saved model
    """
    from cellpose import models, train
    
    os.makedirs(save_path, exist_ok=True)
    
    print("Loading training data...")
    train_images, train_labels = load_data(train_dir)
    print(f"Loaded {len(train_images)} training images")
    
    test_images, test_labels = None, None
    if test_dir:
        print("Loading test data...")
        test_images, test_labels = load_data(test_dir)
        print(f"Loaded {len(test_images)} test images")
    
    print("Initializing model...")
    model = models.CellposeModel(gpu=True)
    
    print(f"Training for {n_epochs} epochs...")
    model_path, train_losses, test_losses = train.train_seg(
        model.net,
        train_data=train_images,
        train_labels=train_labels,
        test_data=test_images,
        test_labels=test_labels,
        save_path=str(Path(save_path).parent),
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        weight_decay=0.1,
        model_name=model_name
    )
    
    print(f"\nModel saved: {model_path}")
    return model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Cellpose model")
    parser.add_argument("--train", required=True, help="Training data directory")
    parser.add_argument("--test", default=None, help="Test data directory (optional)")
    parser.add_argument("--name", default="polystyrene_model", help="Model name")
    parser.add_argument("--save", default="./models", help="Save directory")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    args = parser.parse_args()
    
    train_model(args.train, args.test, args.name, args.save, args.epochs, args.lr)
