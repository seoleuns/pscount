"""
Predict and visualize particle counts using trained model.
"""
import numpy as np
import tifffile
import cv2
from pathlib import Path
import argparse
import os
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt


def predict_and_visualize(model_path, image_dir, output_dir, save_images=True):
    """
    Predict particle counts and optionally save visualizations.
    
    Args:
        model_path: Path to trained Cellpose model
        image_dir: Directory with images to predict
        output_dir: Directory for output visualizations
        save_images: Whether to save visualization images
    
    Returns:
        List of results dictionaries
    """
    from cellpose import models
    
    if save_images:
        os.makedirs(output_dir, exist_ok=True)
    
    print("Loading model...")
    if model_path == "default":
            model = models.CellposeModel(gpu=True)  
    else:
        model = models.CellposeModel(gpu=True, pretrained_model=model_path)
    
    image_dir = Path(image_dir)
    tif_files = sorted(image_dir.glob("*.tif"))
    
    if not tif_files:
        print(f"No TIF files found in {image_dir}")
        return []
    
    results = []
    
    for tif_file in tif_files:
        img = tifffile.imread(str(tif_file))
        if model_path == "default":
           masks, flows, _ = model.eval(img)
           #masks, flows, _ = model.eval(img,diameter=16) # the parameter optimized for the default Cellpose
        else:
           masks, flows, _ = model.eval(img)
        pred_count = int(masks.max())
        
        results.append({
            'filename': tif_file.name,
            'count': pred_count
        })
        
        print(f"{tif_file.name}: {pred_count} particles")
        
        if save_images:
            # Visualization
            if img.max() > 255:
                img_vis = (img / img.max() * 255).astype(np.uint8)
            else:
                img_vis = img.astype(np.uint8)
            
            if len(img_vis.shape) == 2:
                img_vis = cv2.cvtColor(img_vis, cv2.COLOR_GRAY2BGR)
            
            for i in range(1, pred_count + 1):
                mask_i = (masks == i).astype(np.uint8)
                contours, _ = cv2.findContours(mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img_vis, contours, -1, (0, 255, 0), 1)
                
                M = cv2.moments(mask_i)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(img_vis, str(i), (cx-5, cy+5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
            
            output_path = Path(output_dir) / f"{tif_file.stem}_result.png"
            cv2.imwrite(str(output_path), img_vis)
            #Probability Mapping
            cellprob = flows[2]
            cellprob_normalized = 1 / (1 + np.exp(-cellprob))
            plt.figure()
            plt.imshow(cellprob_normalized, cmap='hot')
            plt.clim(0, 1)
            plt.axis("off")
            plt.tight_layout()
            plt.colorbar()
            output_path = Path(output_dir) / f"{tif_file.stem}_result_prob_map.png"
            plt.savefig(output_path)
            plt.close()
    
    print(f"\nTotal: {len(results)} images processed")
    if save_images:
        print(f"Visualizations saved to {output_dir}")
    
    return results

def evaluate_with_ground_truth(model_path, image_dir, labeled_dir, output_dir, save_images=True):
    """..."""
    import re
    import pandas as pd
    from cellpose import models
    
    os.makedirs(output_dir, exist_ok=True)

    print("Loading model...")
    if model_path == "default":
        print("Default CellPose model...")
        model = models.CellposeModel(gpu=True)
    else:
        print("Trained model...")
        model = models.CellposeModel(gpu=True, pretrained_model=model_path)
    
    labeled_files = sorted(Path(labeled_dir).glob("*.tif"))
    results = []
    
    for labeled_path in labeled_files:
        labeled_name = labeled_path.name
        
        match = re.search(r'_(\d+)count_Flatten', labeled_name)
        if not match:
            continue
        true_count = int(match.group(1))
        
        original_name = re.sub(r'_\d+count_Flatten', '', labeled_name)
        img_path = Path(image_dir) / original_name
        
        if not img_path.exists():
            continue
        
        img = tifffile.imread(str(img_path))
        if model_path == "default":
           masks, flows, _ = model.eval(img)
           #masks, flows, _ = model.eval(img,diameter=16) # the parameter optimized for the default Cellpose
        else:
           masks, flows, _ = model.eval(img)
        pred_count = int(masks.max())
        
        error = pred_count - true_count
        results.append({
            'filename': original_name,
            'true': true_count,
            'pred': pred_count,
            'error': error
        })
        
        print(f"{original_name}: true={true_count}, pred={pred_count}, error={error:+d}")
        
        if save_images:
            # Visualization
            if img.max() > 255:
                img_vis = (img / img.max() * 255).astype(np.uint8)
            else:
                img_vis = img.astype(np.uint8)
            
            if len(img_vis.shape) == 2:
                img_vis = cv2.cvtColor(img_vis, cv2.COLOR_GRAY2BGR)
            
            for i in range(1, pred_count + 1):
                mask_i = (masks == i).astype(np.uint8)
                contours, _ = cv2.findContours(mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img_vis, contours, -1, (0, 255, 0), 1)
                
                M = cv2.moments(mask_i)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(img_vis, str(i), (cx-5, cy+5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
            
            cv2.imwrite(str(Path(output_dir) / f"{img_path.stem}_result.png"), img_vis)
            
            # Probability Mapping
            cellprob = flows[2]
            cellprob_normalized = 1 / (1 + np.exp(-cellprob))
            plt.figure()
            plt.imshow(cellprob_normalized, cmap='hot')
            plt.clim(0, 1)
            plt.axis("off")
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(Path(output_dir) / f"{img_path.stem}_prob_map.png")
            plt.close()
    
    results_df = pd.DataFrame(results)
    mae = results_df['error'].abs().mean()
    mae = round(mae, 1) 
    print(f"MAE: {mae}")
    
    results_df.to_csv(Path(output_dir) / "evaluation_results.csv", index=False)
    
    return results_df

def evaluate_with_csv(model_path, image_dir, csv_path, output_dir, save_images=True):
    """
    Evaluate model with ground truth from CSV file.
    
    CSV format: filename,count
    """
    import pandas as pd
    from cellpose import models
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading model...")
    if model_path == "default":
        model = models.CellposeModel(gpu=True)
    else:
        model = models.CellposeModel(pretrained_model=model_path, gpu=True)
    
    df = pd.read_csv(csv_path)
    results = []
    
    for _, row in df.iterrows():
        filename = row['filename']
        true_count = int(row['count'])
        
        img_path = Path(image_dir) / filename
        if not img_path.exists():
            print(f"Not found: {img_path}")
            continue
        
        img = tifffile.imread(str(img_path))
        if model_path == "default":
           masks, flows, _ = model.eval(img)
           #masks, flows, _ = model.eval(img,diameter=16)
        else:
           masks, flows, _ = model.eval(img)
        pred_count = int(masks.max())
        
        error = pred_count - true_count
        results.append({
            'filename': filename,
            'true': true_count,
            'pred': pred_count,
            'error': error
        })
        
        print(f"{filename}: true={true_count}, pred={pred_count}, error={error:+d}")
    
        if save_images:
            # Visualization
            if img.max() > 255:
                img_vis = (img / img.max() * 255).astype(np.uint8)
            else:
                img_vis = img.astype(np.uint8)
            
            if len(img_vis.shape) == 2:
                img_vis = cv2.cvtColor(img_vis, cv2.COLOR_GRAY2BGR)
            
            for i in range(1, pred_count + 1):
                mask_i = (masks == i).astype(np.uint8)
                contours, _ = cv2.findContours(mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img_vis, contours, -1, (0, 255, 0), 1)
                
                M = cv2.moments(mask_i)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(img_vis, str(i), (cx-5, cy+5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
            
            cv2.imwrite(str(Path(output_dir) / f"{img_path.stem}_result.png"), img_vis)
            
            # Probability Mapping
            cellprob = flows[2]
            cellprob_normalized = 1 / (1 + np.exp(-cellprob))
            plt.figure()
            plt.imshow(cellprob_normalized, cmap='hot')
            plt.colorbar()
            plt.clim(0, 1)
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(Path(output_dir) / f"{img_path.stem}_prob_map.png")
            plt.close()

    results_df = pd.DataFrame(results)
    mae = results_df['error'].abs().mean()
    print(f"\nMAE: {mae:.1f}")
    
    results_df.to_csv(Path(output_dir) / "evaluation_results.csv", index=False)
    
    return results_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict particle counts")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--input", required=True, help="Input image directory")
    parser.add_argument("--output", default="./results", help="Output directory")
    parser.add_argument("--test-labeled", default=None, help="Labeled directory for evaluation")
    parser.add_argument("--no-images", action="store_true", help="Skip saving visualizations")
    args = parser.parse_args()

    if args.labeled:
        evaluate_with_ground_truth(args.model, args.input, args.labeled, args.output)
    else:
        predict_and_visualize(args.model, args.input, args.output, not args.no_images)
