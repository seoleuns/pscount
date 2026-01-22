"""
Create Cellpose training masks from point coordinates.
"""
import re
import numpy as np
from pathlib import Path
import pandas as pd
import tifffile
import argparse
import os
import cv2



def create_mask_from_points(image_shape, points, radius=8):
    """
    Create a mask from point coordinates.
    
    Args:
        image_shape: Shape of the image (H, W) or (H, W, C)
        points: List of (x, y) coordinates
        radius: Radius of each cell mask
    
    Returns:
        Mask array with unique labels for each cell
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint16)
    
    for i, (x, y) in enumerate(points):
        yy, xx = np.ogrid[:image_shape[0], :image_shape[1]]
        circle = (xx - x)**2 + (yy - y)**2 <= radius**2
        circle = circle & (mask == 0)
        mask[circle] = i + 1
    
    return mask


def create_masks_from_csv(csv_path, original_dir, output_dir, radius=8):
    """
    Create training masks from CSV coordinates.
    
    Args:
        csv_path: CSV file with filename and coordinates
        original_dir: Directory with original images
        output_dir: Output directory for training data
        radius: Radius for each cell mask
    """
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(csv_path)
    
    for idx, row in df.iterrows():
        label_filename = row['filename']
        coords_str = row['coordinates']
        
        # Extract original filename: remove _NUMcount_Flatten
        original_filename = re.sub(r'_\d+count_Flatten', '', label_filename)
        
        # Parse coordinates
        points = []
        if pd.notna(coords_str) and str(coords_str).strip():
            for coord in str(coords_str).split(';'):
                if ',' in coord:
                    x, y = map(int, coord.split(','))
                    points.append((x, y))
        
        original_path = Path(original_dir) / original_filename
        
        if not original_path.exists():
            print(f"Not found: {original_path}")
            continue
        
        img = tifffile.imread(str(original_path))
        mask = create_mask_from_points(img.shape, points, radius)
        
        if img.max() > 255:
            img_vis = (img / img.max() * 255).astype(np.uint8)
        else:
            img_vis = img.astype(np.uint8)

        if len(img_vis.shape) == 2:
            img_vis = cv2.cvtColor(img_vis, cv2.COLOR_GRAY2BGR)

        for i in range(1, int(mask.max()) + 1):
            mask_i = (mask == i).astype(np.uint8)
            contours, _ = cv2.findContours(mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img_vis, contours, -1, (0, 255, 0), 1)

        base_name = Path(original_filename).stem
        tifffile.imwrite(str(Path(output_dir) / f"{base_name}.tif"), img)
        cv2.imwrite(str(Path(output_dir) / f"{base_name}_mask_check.png"), img_vis)
        np.save(str(Path(output_dir) / f"{base_name}_masks.npy"), mask)
        
        print(f"{original_filename}: {len(points)} cells -> mask saved")
    
    print(f"\nDone: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create training masks from point coordinates")
    parser.add_argument("--csv", required=True, help="CSV file with coordinates")
    parser.add_argument("--original", required=True, help="Directory with original images")
    parser.add_argument("--output", required=True, help="Output directory for training data")
    parser.add_argument("--radius", type=int, default=8, help="Mask radius (default: 8)")
    args = parser.parse_args()
    
    create_masks_from_csv(args.csv, args.original, args.output, args.radius)
