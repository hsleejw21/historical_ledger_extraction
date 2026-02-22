"""
experiments/v6_loocv/extract_visual_features.py

Extracts comprehensive visual features from historical ledger images.
These features serve as predictors for the adaptive routing system in Phase 3.

Features extracted (20+ per image):
  1. Dimensions: height, width, aspect_ratio, total_pixels
  2. Brightness: mean, std, median, min, max
  3. Contrast: dynamic_range, contrast_ratio
  4. Edge features: edge_density, edge_count, avg_edge_strength
  5. Text regions: text_block_count, avg_block_size, block_density
  6. Layout: estimated_columns
  7. Quality: blur_estimate, noise_estimate, sharpness
  8. Histogram: entropy, skewness, dominant_brightness_bin

Usage:
    python -m experiments.v6_loocv.extract_visual_features [--visualize]
    
    --visualize: Generate feature distribution plots (optional)
"""
import os
import sys
import json
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from scipy import stats
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import DATA_DIR, BASE_DIR


# Output path
FEATURES_OUTPUT_DIR = os.path.join(BASE_DIR, "data", "visual_features")
FEATURES_OUTPUT_PATH = os.path.join(FEATURES_OUTPUT_DIR, "visual_features.json")


def extract_dimension_features(img_array):
    """Extract basic dimension and size features."""
    height, width = img_array.shape[:2]
    
    return {
        'height': int(height),
        'width': int(width),
        'aspect_ratio': round(height / width, 3),
        'total_pixels': int(height * width),
        'megapixels': round(height * width / 1e6, 2),
    }


def extract_brightness_features(img_gray):
    """Extract brightness statistics from grayscale image."""
    pixels = img_gray.flatten()
    
    return {
        'brightness_mean': round(float(np.mean(pixels)), 2),
        'brightness_std': round(float(np.std(pixels)), 2),
        'brightness_median': round(float(np.median(pixels)), 2),
        'brightness_min': int(np.min(pixels)),
        'brightness_max': int(np.max(pixels)),
    }


def extract_contrast_features(img_gray):
    """Extract contrast and dynamic range features."""
    pixels = img_gray.flatten()
    
    max_val = float(np.max(pixels))
    min_val = float(np.min(pixels))
    dynamic_range = int(max_val - min_val)
    
    # Michelson contrast: (max - min) / (max + min)
    if max_val + min_val > 0:
        contrast_ratio = round((max_val - min_val) / (max_val + min_val), 3)
    else:
        contrast_ratio = 0.0
    
    # RMS contrast
    rms_contrast = round(float(np.std(pixels)), 2)
    
    return {
        'dynamic_range': dynamic_range,
        'contrast_ratio': contrast_ratio,
        'rms_contrast': rms_contrast,
    }


def extract_edge_features(img_gray):
    """Extract edge-based features using Canny edge detection."""
    # Canny edge detection with auto-threshold
    sigma = 0.33
    median_val = np.median(img_gray)
    lower = int(max(0, (1.0 - sigma) * median_val))
    upper = int(min(255, (1.0 + sigma) * median_val))
    
    edges = cv2.Canny(img_gray, lower, upper)
    
    # Edge statistics
    edge_pixels = np.sum(edges > 0)
    total_pixels = edges.shape[0] * edges.shape[1]
    edge_density = edge_pixels / total_pixels
    
    # Average edge strength (from Sobel)
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
    avg_edge_strength = round(float(np.mean(edge_magnitude)), 2)
    
    return {
        'edge_density': round(edge_density, 5),
        'edge_count': int(edge_pixels),
        'avg_edge_strength': avg_edge_strength,
    }


def extract_text_block_features(img_gray):
    """Extract text region features using connected components."""
    # Otsu's thresholding to binarize
    _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphological operations to connect text regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(binary, kernel, iterations=2)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    
    # Filter small components (noise)
    min_area = 100
    valid_components = stats[1:, cv2.CC_STAT_AREA] >= min_area  # Skip background (label 0)
    
    text_block_count = int(np.sum(valid_components))
    
    if text_block_count > 0:
        valid_areas = stats[1:][valid_components, cv2.CC_STAT_AREA]
        avg_block_size = round(float(np.mean(valid_areas)), 2)
        block_area_ratio = round(float(np.sum(valid_areas)) / (img_gray.shape[0] * img_gray.shape[1]), 4)
    else:
        avg_block_size = 0.0
        block_area_ratio = 0.0
    
    return {
        'text_block_count': text_block_count,
        'avg_block_size': avg_block_size,
        'block_area_ratio': block_area_ratio,
    }


def extract_layout_features(img_gray, edges):
    """Extract layout features - column detection, ruling lines."""
    height, width = img_gray.shape
    
    # Estimate column count based on vertical line density
    # Project edges onto x-axis
    vertical_projection = np.sum(edges, axis=0)
    
    # Find peaks in projection (potential column boundaries)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(vertical_projection, height=np.max(vertical_projection) * 0.3, distance=width // 10)
    estimated_columns = len(peaks) + 1  # Number of columns = number of dividers + 1
    
    return {
        'estimated_columns': int(estimated_columns),
    }


def extract_quality_features(img_gray):
    """Extract image quality features - blur, noise, sharpness."""
    # Blur estimate using Laplacian variance
    laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
    blur_estimate = round(float(laplacian.var()), 2)
    
    # Noise estimate using high-frequency content
    # Apply Gaussian blur and compute difference
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    noise_map = cv2.absdiff(img_gray, blurred)
    noise_estimate = round(float(np.mean(noise_map)), 2)
    
    # Sharpness using gradient magnitude
    gx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gx**2 + gy**2)
    sharpness = round(float(np.mean(gradient_magnitude)), 2)
    
    return {
        'blur_estimate': blur_estimate,
        'noise_estimate': noise_estimate,
        'sharpness': sharpness,
    }


def extract_histogram_features(img_gray):
    """Extract histogram-based features - entropy, skewness."""
    pixels = img_gray.flatten()
    
    # Histogram entropy (measure of information content)
    hist, _ = np.histogram(pixels, bins=256, range=(0, 256))
    hist_normalized = hist / np.sum(hist)
    hist_normalized = hist_normalized[hist_normalized > 0]  # Remove zeros for log
    entropy = round(float(-np.sum(hist_normalized * np.log2(hist_normalized))), 3)
    
    # Skewness of brightness distribution
    skewness = round(float(stats.skew(pixels)), 3)
    
    # Dominant brightness bin (mode)
    dominant_bin = int(np.argmax(hist))
    
    # Brightness percentiles
    p25 = round(float(np.percentile(pixels, 25)), 2)
    p75 = round(float(np.percentile(pixels, 75)), 2)
    
    return {
        'histogram_entropy': entropy,
        'brightness_skewness': skewness,
        'dominant_brightness_bin': dominant_bin,
        'brightness_p25': p25,
        'brightness_p75': p75,
    }


def extract_all_features(image_path):
    """Extract all visual features from a single image."""
    # Load image
    img = Image.open(image_path)
    img_array = np.array(img)
    
    # Convert to grayscale
    if len(img_array.shape) == 3:
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img_array
    
    # Extract Canny edges (used by multiple feature extractors)
    sigma = 0.33
    median_val = np.median(img_gray)
    lower = int(max(0, (1.0 - sigma) * median_val))
    upper = int(min(255, (1.0 + sigma) * median_val))
    edges = cv2.Canny(img_gray, lower, upper)
    
    # Combine all features
    features = {}
    
    features.update(extract_dimension_features(img_array))
    features.update(extract_brightness_features(img_gray))
    features.update(extract_contrast_features(img_gray))
    features.update(extract_edge_features(img_gray))
    features.update(extract_text_block_features(img_gray))
    features.update(extract_layout_features(img_gray, edges))
    features.update(extract_quality_features(img_gray))
    features.update(extract_histogram_features(img_gray))
    
    return features


def extract_features_for_all_images(image_dir):
    """Extract features for all images in the directory."""
    print("\n" + "="*60)
    print("PHASE 2: Visual Feature Extraction")
    print("="*60)
    
    # Discover all PNG images
    image_files = list(Path(image_dir).glob("*.png"))
    
    if not image_files:
        print(f"\n[Error] No PNG images found in {image_dir}")
        return {}
    
    print(f"\n[1/2] Found {len(image_files)} images")
    print(f"      Directory: {image_dir}")
    
    # Extract features
    print(f"\n[2/2] Extracting features...")
    
    all_features = {}
    
    for img_path in tqdm(image_files, desc="Processing images", unit="img"):
        # Extract page name from filename
        page_name = img_path.stem  # e.g., "1700_7" or "1700_7_image"
        if page_name.endswith("_image"):
            page_name = page_name[: -len("_image")]
        
        try:
            features = extract_all_features(str(img_path))
            all_features[page_name] = features
        except Exception as e:
            print(f"\n  [Error] {page_name}: {e}")
            continue
    
    print(f"\n[OK] Extracted features for {len(all_features)}/{len(image_files)} images")
    
    return all_features


def save_features(features, output_path):
    """Save features to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(features, f, indent=2, ensure_ascii=False)
    
    print(f"\n[Saved] {output_path}")


def print_feature_summary(features):
    """Print summary statistics of extracted features."""
    if not features:
        return
    
    print("\n" + "="*60)
    print("FEATURE SUMMARY")
    print("="*60)
    
    # Get all feature keys from first page
    first_page = list(features.values())[0]
    feature_keys = first_page.keys()
    
    print(f"\nTotal pages: {len(features)}")
    print(f"Features per page: {len(feature_keys)}")
    
    print(f"\nFeature list:")
    for i, key in enumerate(sorted(feature_keys), 1):
        print(f"  {i:2d}. {key}")
    
    # Compute min/max/avg for numeric features
    print(f"\nFeature ranges:")
    
    for key in sorted(feature_keys):
        values = [f[key] for f in features.values() if isinstance(f[key], (int, float))]
        
        if values:
            min_val = min(values)
            max_val = max(values)
            avg_val = sum(values) / len(values)
            print(f"  {key:30s}: min={min_val:>8.2f}  max={max_val:>8.2f}  avg={avg_val:>8.2f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract visual features from ledger images")
    parser.add_argument(
        "--image-dir",
        default=DATA_DIR,
        help="Directory containing images (default: data/images/)"
    )
    parser.add_argument(
        "--output",
        default=FEATURES_OUTPUT_PATH,
        help="Output JSON path (default: data/visual_features/visual_features.json)"
    )
    
    args = parser.parse_args()
    
    # Extract features
    features = extract_features_for_all_images(args.image_dir)
    
    if features:
        # Save to JSON
        save_features(features, args.output)
        
        # Print summary
        print_feature_summary(features)
        
        print("\n" + "="*60)
        print("Next step: Visualize features")
        print("  python -m experiments.v6_loocv.visualize_features")
        print("="*60 + "\n")
    else:
        print("\n[Error] No features extracted. Check image directory and file format.")