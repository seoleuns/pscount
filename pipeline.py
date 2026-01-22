"""
End-to-end pipeline for polystyrene particle counting.

This pipeline:
1. Extracts red point coordinates from labeled images
2. Creates training masks from coordinates
3. Fine-tunes Cellpose model 
4. Predicts particle counts on test images
"""
import argparse
import os
from pathlib import Path

from extract_points import process_directory as extract_points
from create_masks import create_masks_from_csv
from train import train_model
from predict import predict_and_visualize, evaluate_with_ground_truth, evaluate_with_csv


def run_pipeline(
    labeled_dir=None,
    original_dir=None,
    output_dir="./output",
    test_labeled_dir=None,
    test_original_dir=None,
    input_dir=None,
    csv_path=None,
    pretrained_model=None,
    radius=8,
    n_epochs=100,
    model_name="polystyrene_model"
):
    """
    Run the pipeline.
    
    Args:
        labeled_dir: Directory with labeled images for training (red points)
        original_dir: Directory with original images for training
        output_dir: Output directory for all results
        test_labeled_dir: Directory with test labeled images (Evalauation)
        test_original_dir: Directory with test original images (Evalauation)
        input_dir: Directory with images for counting (for all prediction test) 
        pretrained_model: Path to pre-trained model (skip training)
        radius: Radius for mask generation
        n_epochs: Number of training epochs
        model_name: Name for the trained model
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results_dir = output_path 
    
   # Mode 1: Evaluation When Ground Truth (Red points) is available.
    if pretrained_model and test_labeled_dir and test_original_dir:
        print("=" * 50)
        print("Mode: Evaluation with Test Set A")
        print("=" * 50)
        evaluate_with_ground_truth(
            pretrained_model,
            test_original_dir,
            test_labeled_dir,
            str(results_dir)
        )
        print(f"\nResults saved: {results_dir}")
        return

    # Mode 2: Evaluation When Ground Truth (Total Counts saved in CSV format) is available.
    if pretrained_model and input_dir and csv_path:
        print("=" * 50)
        print("Mode: Evaluation with Test Set B")
        print("=" * 50)
        evaluate_with_csv(pretrained_model, input_dir, csv_path, str(results_dir))
        print(f"\nResults saved: {results_dir}")
        return 

    # Mode 3: Prediction only with trained model (Without Ground Truth)
    if pretrained_model and input_dir:
        print("=" * 50)
        print("Mode: Prediction only")
        print("=" * 50)
        predict_and_visualize(pretrained_model, input_dir, str(results_dir))
        print(f"\nResults saved: {results_dir}")
        return
    
    # Mode 4: Full pipeline (training + evaluation)
    if not labeled_dir or not original_dir:
        raise ValueError("--labeled and --original required for training")
    
    csv_path = output_path / "coordinates.csv"
    train_data_dir = output_path / "train_data"
    test_data_dir = output_path / "test_data" if test_labeled_dir else None
    model_dir = output_path / "models"
    
    # Step 1: Extract red points
    print("=" * 50)
    print("Step 1: Extracting red point coordinates")
    print("=" * 50)
    extract_points(labeled_dir, str(csv_path))
    
    # Step 2: Create masks
    print("\n" + "=" * 50)
    print("Step 2: Creating training masks")
    print("=" * 50)
    create_masks_from_csv(str(csv_path), original_dir, str(train_data_dir), radius)
    
    # Step 2b: Create test masks if provided
    if test_labeled_dir and test_original_dir:
        print("\nCreating test masks...")
        test_csv = output_path / "test_coordinates.csv"
        extract_points(test_labeled_dir, str(test_csv))
        create_masks_from_csv(str(test_csv), test_original_dir, str(test_data_dir), radius)
    
    # Step 3: Train or use pre-trained model
    if pretrained_model:
        print("\n" + "=" * 50)
        print("Step 3: Using pre-trained model")
        print("=" * 50)
        model_path = pretrained_model
        print(f"Model: {model_path}")
    else:
        print("\n" + "=" * 50)
        print("Step 3: Training model")
        print("=" * 50)
        model_path = train_model(
            str(train_data_dir),
            str(test_data_dir) if test_data_dir else None,
            model_name,
            str(model_dir),
            n_epochs
        )
    
    # Step 4: Evaluate
    print("\n" + "=" * 50)
    print("Step 4: Evaluation")
    print("=" * 50)
    
    if test_labeled_dir and test_original_dir:
        evaluate_with_ground_truth(
            model_path,
            test_original_dir,
            test_labeled_dir,
            str(results_dir)
        )
    else:
        predict_and_visualize(model_path, str(train_data_dir), str(results_dir))
    
    print("\n" + "=" * 50)
    print("Pipeline complete!")
    print("=" * 50)
    print(f"Model: {model_path}")
    print(f"Results: {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline for polystyrene particle counting"
    )
    
    # Training mode
    parser.add_argument("--labeled", default=None,
                        help="Directory with labeled images (red points)")
    parser.add_argument("--csv", default=None,
                        help="CSV file with ground truth (filename,count)")
    parser.add_argument("--original", default=None,
                        help="Directory with original images")
    parser.add_argument("--test-labeled", default=None,
                        help="Directory with test labeled images")
    parser.add_argument("--test-original", default=None,
                        help="Directory with test original images")
    
    # Prediction mode
    parser.add_argument("--model", default=None,
                        help="Pre-trained model path (skip training)")
    parser.add_argument("--input", default=None,
                        help="Input images for prediction only")
    
    # Common options
    parser.add_argument("--output", default="./output",
                        help="Output directory")
    parser.add_argument("--radius", type=int, default=8,
                        help="Mask radius (default: 8)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Training epochs (default: 100)")
    parser.add_argument("--name", default="polystyrene_model",
                        help="Model name")
    
    args = parser.parse_args()
    
    run_pipeline(
        labeled_dir=args.labeled,
        original_dir=args.original,
        output_dir=args.output,
        test_labeled_dir=args.test_labeled,
        test_original_dir=args.test_original,
        input_dir=args.input,
        csv_path=args.csv,
        pretrained_model=args.model,
        radius=args.radius,
        n_epochs=args.epochs,
        model_name=args.name
    )
