"""
estimate_mask_radius.py

Estimates the appropriate mask radius for PSCount by measuring
the apparent diameter of isolated single particles in a brightfield
microscopy image.

Usage:
    python estimate_mask_radius.py --image image.tif
    python estimate_mask_radius.py --image image.tif --visualize
    python estimate_mask_radius.py --image image.tif --min_area 100 --max_area 500 --circularity 0.75 --visualize

"""

import cv2
import numpy as np
import argparse


def compute_circularity(contour):
    """
    Compute circularity of a contour.
    Perfect circle = 1.0, elongated or irregular shapes < 1.0
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0
    return 4 * np.pi * area / (perimeter ** 2)


def estimate_mask_radius(image_path, min_area=100, max_area=500,
                         min_circularity=0.75, visualize=False):
    """
    Estimate mask radius from isolated single particles in a brightfield image.

    Parameters
    ----------
    image_path : str
        Path to the brightfield microscopy image (tif, png, jpg, etc.)
    min_area : int
        Minimum particle area in pixels (default: 100, ~11px diameter).
        This threshold is adjusted to separate real particles from background noise.
    max_area : int
        Maximum particle area in pixels (default: 500, ~25px diameter).
        Adjust to exclude large clusters.
    min_circularity : float
        Minimum circularity to accept as a single particle (default: 0.75).
        Range 0-1; single spherical particles are close to 1.0,
        clusters and irregular shapes are lower.
        Adjust based on your particle shape.
    visualize : bool
        If True, saves an annotated image showing detected particles.

    Returns
    -------
    dict with median_diameter, mean_diameter, std_diameter,
    recommended_radius, and count
    """

    # Load image as grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    print(f"Image size: {img.shape[1]} x {img.shape[0]} pixels")

    # Adaptive threshold to detect dark particles on bright background.
    binary = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=51,  # neighborhood size; increase for larger particles
        C=5
    )

    # Find contours of connected pixel regions
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 1: Filter by area to get rough single-particle candidates
    area_filtered = [c for c in contours if min_area < cv2.contourArea(c) < max_area]

    # Step 2: Filter by circularity to exclude clusters and irregular shapes.
    single_particles = [c for c in area_filtered if compute_circularity(c) >= min_circularity]

    if len(single_particles) == 0:
        print("No particles found. Try adjusting min_area, max_area, or min_circularity.")
        return None

    # Calculate equivalent diameter: d = 2 * sqrt(area / pi)
    diameters = np.array([2 * np.sqrt(cv2.contourArea(c) / np.pi) for c in single_particles])

    results = {
        "count": len(diameters),
        "median_diameter": np.median(diameters),
        "mean_diameter": np.mean(diameters),
        "std_diameter": np.std(diameters),
        "recommended_radius": round(np.median(diameters) / 2)
    }

    print(f"\nDetected {results['count']} isolated single particles "
          f"(after area and circularity filtering)")
    print(f"Apparent diameter: median = {results['median_diameter']:.1f} px, "
          f"mean = {results['mean_diameter']:.1f} +/- {results['std_diameter']:.1f} px")
    print(f"\n-> Recommended mask radius: {results['recommended_radius']} pixels")

    # Optional: save annotated image for visual verification
    if visualize:
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for c in single_particles:
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                r = int(results["recommended_radius"])
                cv2.circle(vis, (cx, cy), r, (0, 0, 255), 1)
        out_path = image_path.rsplit(".", 1)[0] + "_radius_check.png"
        cv2.imwrite(out_path, vis)
        print(f"Annotated image saved to: {out_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Estimate PSCount mask radius from a brightfield microscopy image."
    )
    parser.add_argument("--image", required=True,
                        help="Path to brightfield microscopy image")
    parser.add_argument("--min_area", type=int, default=100,
                        help="Minimum particle area in pixels (default: 100)")
    parser.add_argument("--max_area", type=int, default=500,
                        help="Maximum particle area in pixels (default: 500)")
    parser.add_argument("--circularity", type=float, default=0.75,
                        help="Minimum circularity threshold (default: 0.75, range: 0-1)")
    parser.add_argument("--visualize", action="store_true",
                        help="Save annotated image showing detected particles")
    args = parser.parse_args()

    estimate_mask_radius(
        args.image,
        args.min_area,
        args.max_area,
        args.circularity,
        args.visualize
    )
