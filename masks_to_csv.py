"""
Convert PNG mask predictions to Kaggle submission CSV with RLE encoding

Usage:
    python src/masks_to_csv.py --mask_dir inference_output/ --output_csv submission.csv
"""

import argparse
import csv
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image
from tqdm import tqdm


def encode_rle(mask: np.ndarray) -> str:
    """
    Encode binary mask to Run-Length Encoding (RLE) in column-major (Fortran) order.
    
    Args:
        mask: Binary mask (H, W) with values {0, 1} or {0, 255}
        
    Returns:
        RLE string (space-separated run lengths)
    """
    # Convert to binary {0, 1} if needed
    if mask.max() > 1:
        mask = (mask > 127).astype(np.uint8)
    
    # Flatten in column-major order (Fortran order)
    flat_mask = mask.flatten(order='F')
    
    # Run-length encode
    runs = []
    i = 0
    while i < len(flat_mask):
        run_value = flat_mask[i]
        run_length = 1
        
        # Count consecutive values
        while i + run_length < len(flat_mask) and flat_mask[i + run_length] == run_value:
            run_length += 1
        
        runs.append(run_length)
        i += run_length
    
    # Convert runs to string, space-separated
    rle_string = ' '.join(map(str, runs))
    return rle_string


def decode_rle(rle_string: str, height: int, width: int) -> np.ndarray:
    """
    Decode RLE string back to binary mask.
    
    Args:
        rle_string: RLE encoded string (space-separated run lengths)
        height: Mask height
        width: Mask width
        
    Returns:
        Binary mask (H, W) with values {0, 1}
    """
    if isinstance(rle_string, str):
        runs = list(map(int, rle_string.split()))
    else:
        runs = rle_string
    
    # Decode RLE
    decoded = []
    value = 0  # Start with 0
    for run_length in runs:
        decoded.extend([value] * run_length)
        value = 1 - value  # Toggle between 0 and 1
    
    # Reshape from flattened column-major to (H, W)
    mask = np.array(decoded, dtype=np.uint8).reshape((height, width), order='F')
    return mask


def masks_to_csv(mask_dir: str, output_csv: str, mask_suffix: str = "_mask.png") -> None:
    """
    Convert PNG masks to Kaggle submission CSV with RLE encoding.
    
    Args:
        mask_dir: Directory containing PNG masks
        output_csv: Path to output CSV file
        mask_suffix: Suffix of mask files (e.g., "_mask.png")
    """
    mask_dir = Path(mask_dir)
    
    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")
    
    # Find all mask PNG files
    mask_files = sorted(mask_dir.glob(f"*{mask_suffix}"))
    
    if len(mask_files) == 0:
        raise ValueError(f"No masks found in {mask_dir} with suffix '{mask_suffix}'")
    
    print(f"Found {len(mask_files)} masks in {mask_dir}")
    
    # Create CSV with RLE encoding
    print(f"Creating Kaggle submission: {output_csv}")
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_id', 'encoded_mask'])
        
        for mask_path in tqdm(mask_files, desc="Encoding RLE"):
            # Load mask
            mask = np.array(Image.open(mask_path))
            
            # Extract image_id (remove suffix)
            image_id = mask_path.stem.replace(mask_suffix.replace('.png', ''), '')
            
            # Encode RLE
            rle_string = encode_rle(mask)
            
            # Write to CSV
            writer.writerow([image_id, rle_string])
    
    print(f"Submission saved to {output_csv}")
    print(f"Total predictions: {len(mask_files)}")
    
    # Verify CSV format
    print("\nCSV preview (first 3 rows):")
    with open(output_csv, 'r') as f:
        for i, line in enumerate(f):
            if i < 3:
                print(f"  {line.strip()}")
            else:
                break


def main():
    parser = argparse.ArgumentParser(description="Convert PNG masks to Kaggle submission CSV")
    parser.add_argument(
        '--mask_dir',
        type=str,
        required=True,
        help='Directory containing PNG mask predictions'
    )
    parser.add_argument(
        '--output_csv',
        type=str,
        required=True,
        help='Path to output CSV file for Kaggle submission'
    )
    parser.add_argument(
        '--mask_suffix',
        type=str,
        default='_mask.png',
        help='Suffix of mask files (default: _mask.png)'
    )
    
    args = parser.parse_args()
    
    masks_to_csv(
        mask_dir=args.mask_dir,
        output_csv=args.output_csv,
        mask_suffix=args.mask_suffix,
    )


if __name__ == '__main__':
    main()
