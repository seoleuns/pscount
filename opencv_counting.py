# ----------------------------------------------------------------------------------------- 
# OpenCV-based Particle Counting (Baseline comparison for PSCount)
# 
# Original: Seoleun Shin, KRISS, July 20, 2025.
# Modified: TensorFlow dependencies removed for PSCount environment
#------------------------------------------------------------------------------------------
import os 
import glob 
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tifffile
import pandas as pd
from pathlib import Path


def load_images_from_dir(img_dir, image_size=None):
    """Load images from directory without TensorFlow."""
    file_list = sorted(glob.glob(os.path.join(img_dir, "*.tif")))
    img_list = []
    filenames = []
    
    for fig in file_list:
        # Load with tifffile (handles 16-bit TIF)
        img = tifffile.imread(fig)
        
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Resize if specified
        if image_size is not None:
            img = cv2.resize(img, (image_size, image_size))
        
        # Normalize to 0-255 uint8
        if img.max() > 255:
            img = (img / img.max() * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        
        img_list.append(img)
        filenames.append(os.path.basename(fig))
    
    return img_list, filenames


def count_objects_opencv(img, min_size=15):
    """
    Count objects using OpenCV contour detection.
    
    Args:
        img: Grayscale image (uint8)
        min_size: Minimum contour area to count
    
    Returns:
        num_objects: Number of detected objects
        processed_mask: Visualization image
    """
    # Threshold using Otsu's method
    _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations to clean up
    kernel = np.ones((1, 1), np.uint8)
    processed_mask = cv2.dilate(bw, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(processed_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Avoid counting the image boundary as a contour
    h, w = processed_mask.shape[:2]
    contours = [cnt for cnt in contours 
            if cv2.contourArea(cnt) < h * w * 0.9]
    
    # Filter by size and remove duplicates
    kept = []
    centers = []
    
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_size:
            continue
        
        # Calculate centroid
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # Check for duplicate (centroid very close to existing)
        if any((cx - x)**2 + (cy - y)**2 < 25 for (x, y) in centers):
            continue
        
        kept.append(c)
        centers.append((cx, cy))
    
    # Create visualization
    vis_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for idx, (cnt, (cx, cy)) in enumerate(zip(kept, centers), start=1):
        cv2.drawContours(vis_img, [cnt], -1, (0, 255, 0), 2)
        #cv2.circle(vis_img, (cx, cy), 2, (0, 0, 255), -1)
        cv2.putText(vis_img, str(idx), (cx + 5, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
    
    return len(kept), vis_img


def evaluate_opencv(image_dir, csv_path=None, labeled_dir=None, output_dir="./results_opencv"):
    """
    Evaluate OpenCV counting method.
    
    Args:
        image_dir: Directory with test images
        csv_path: CSV file with ground truth (filename, count)
        labeled_dir: Directory with labeled images (count in filename)
        output_dir: Output directory for results
    """
    import re
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load images
    images, filenames = load_images_from_dir(image_dir)
    
    # Get ground truth
    if csv_path:
        df_gt = pd.read_csv(csv_path)
        gt_dict = dict(zip(df_gt['filename'], df_gt['count']))
    elif labeled_dir:
        gt_dict = {}
        for f in glob.glob(os.path.join(labeled_dir, "*.tif")):
            name = os.path.basename(f)
            match = re.search(r'_(\d+)count_Flatten', name)
            if match:
                original_name = re.sub(r'_\d+count_Flatten', '', name)
                gt_dict[original_name] = int(match.group(1))
    else:
        gt_dict = {}
    
    # Process each image
    results = []
    
    for img, filename in zip(images, filenames):
        pred_count, vis_img = count_objects_opencv(img)
        
        true_count = gt_dict.get(filename, None)
        
        if true_count is not None:
            error = pred_count - true_count
            results.append({
                'filename': filename,
                'true': true_count,
                'pred': pred_count,
                'error': error
            })
            print(f"{filename}: true={true_count}, pred={pred_count}, error={error:+d}")
        else:
            results.append({
                'filename': filename,
                'pred': pred_count
            })
            print(f"{filename}: pred={pred_count}")
        
        # Save visualization
        output_path = os.path.join(output_dir, f"{Path(filename).stem}_opencv.png")
        cv2.imwrite(output_path, vis_img)
    
    # Calculate MAE if ground truth available
    results_df = pd.DataFrame(results)
    
    if 'true' in results_df.columns:
        mae = results_df['error'].abs().mean()
        print(f"\nOpenCV MAE: {mae:.1f}")
    
    results_df.to_csv(os.path.join(output_dir, "opencv_results.csv"), index=False)
    
    return results_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="OpenCV baseline counting")
    parser.add_argument("--input", required=True, help="Input image directory")
    parser.add_argument("--csv", default=None, help="CSV with ground truth")
    parser.add_argument("--labeled", default=None, help="Labeled image directory")
    parser.add_argument("--output", default="./results_opencv", help="Output directory")
    args = parser.parse_args()
    
    evaluate_opencv(args.input, args.csv, args.labeled, args.output)
