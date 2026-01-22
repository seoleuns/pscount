"""
Extract red point coordinates from labeled images.
"""
import cv2
import numpy as np
from pathlib import Path
import argparse
import tifffile


def extract_red_points(image_path, red_threshold=(150, 100, 100)):
    """
    Extract red point coordinates from a labeled image.
    
    Args:
        image_path: Path to the labeled image
        red_threshold: Tuple of (R_min, G_max, B_max) for red detection
    
    Returns:
        List of (x, y) coordinates
    """
    img = tifffile.imread(str(image_path))
    
    if img is None:
        return []
    
    r_min, g_max, b_max = red_threshold
    red_mask = (img[:,:,0] > r_min) & (img[:,:,1] < g_max) & (img[:,:,2] < b_max)
    red_mask_uint8 = red_mask.astype(np.uint8) * 255
    
    contours, _ = cv2.findContours(red_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    points = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            points.append((cx, cy))
    
    return points


def process_directory(input_dir, output_csv):
    """
    Process all TIF files in directory and save coordinates to CSV.
    
    Args:
        input_dir: Directory containing labeled images
        output_csv: Output CSV file path
    """
    input_path = Path(input_dir)
    tif_files = sorted(input_path.glob("*.tif"))
    
    if not tif_files:
        print(f"No TIF files found in {input_dir}")
        return
    
    print(f"Processing {len(tif_files)} files...")
    
    with open(output_csv, 'w') as f:
        f.write("filename,coordinates\n")
        for img_path in tif_files:
            points = extract_red_points(img_path)
            coords_str = ";".join([f"{x},{y}" for x, y in points])
            f.write(f"{img_path.name},\"{coords_str}\"\n")
            print(f"{img_path.name}: {len(points)} points")
    
    print(f"\nSaved to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract red point coordinates from labeled images")
    parser.add_argument("--input", required=True, help="Input directory with labeled TIF images")
    parser.add_argument("--output", required=True, help="Output CSV file path")
    args = parser.parse_args()
    
    process_directory(args.input, args.output)
