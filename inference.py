"""
Inference script for semantic segmentation - ALL test-time inference logic here

Step 2 of the workflow: Generate predictions and output as Kaggle submission CSV

Usage:
    python src/inference.py \\
      --model unet \\
      --checkpoint saved_models/unet_best.pth \\
      --test_list test_unet.txt \\
      --image_dir dataset/oxford-iiit-pet/images \\
      --output_csv submission.csv
"""

import argparse
import csv
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from tqdm import tqdm

from models.unet import UNet
from models.resnet34_unet import ResNet34UNet
from utils import get_device


def encode_rle(mask: np.ndarray) -> str:
    """
    Encode binary mask to Run-Length Encoding (RLE) in row-major (C) order.
    
    Kaggle format for this competition:
    - Flatten in row-major order
    - Invert: 0 = foreground (pet), 1 = background
    - Alternate run lengths starting with background pixels
    
    Args:
        mask: Binary mask (H, W) with values {0, 1} or {0, 255}
              where 1 (or 255) = foreground, 0 = background
        
    Returns:
        RLE string (space-separated run lengths)
    """
    # Convert to binary {0, 1} if needed
    if mask.max() > 1:
        mask = (mask > 127).astype(np.uint8)
    
    # Invert: convert 1 (foreground) to 0, and 0 (background) to 1
    # This is because Kaggle RLE starts counting from background pixels
    mask_inverted = 1 - mask
    
    # Flatten in row-major order (C order)
    flat_mask = mask_inverted.flatten(order='C')
    
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


class SegmentationInference:
    """
    Inference engine for semantic segmentation.
    
    This is the ONLY class handling test-time inference logic.
    All inference-related operations are centralized here.
    """
    
    def __init__(
        self,
        model: nn.Module,
        checkpoint_path: str,
        device: str = 'cpu',
        image_size: Tuple[int, int] = (256, 256),
    ):
        """
        Initialize inference engine
        
        Args:
            model: PyTorch model
            checkpoint_path: Path to model checkpoint
            device: Device for inference
            image_size: Target image size for preprocessing
        """
        self.model = model.to(device)
        self.device = device
        self.image_size = image_size
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded from {checkpoint_path}")
        
        # Image normalization parameters (ImageNet)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    
    def preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Preprocess image for inference
        
        Args:
            image_path: Path to input image
            
        Returns:
            Tuple of (preprocessed tensor, original image size)
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        original_size = image.size  # (W, H)
        
        # Resize
        image_resized = image.resize(self.image_size, Image.Resampling.BILINEAR)
        
        # Convert to tensor
        image_array = np.array(image_resized).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        
        # Normalize
        image_tensor = (image_tensor - self.mean) / self.std
        
        return image_tensor.to(self.device), original_size
    
    def postprocess_mask(
        self,
        output: torch.Tensor,
        original_size: Tuple[int, int],
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Postprocess model output to binary mask
        
        Args:
            output: Model output (1, 1, H, W)
            original_size: Original image size (W, H) - NOT USED, kept for compatibility
            threshold: Threshold for binarization
            
        Returns:
            Binary mask as numpy array (H, W) with values {0, 1}
            Kaggle requires 256x256 fixed size, so we return model output size (256x256)
        """
        # Apply sigmoid
        pred_sigmoid = torch.sigmoid(output)
        
        # Binarize with threshold
        binary_mask = (pred_sigmoid >= threshold).float()
        
        # Squeeze and move to CPU
        binary_mask = binary_mask.squeeze().cpu().numpy()
        
        # IMPORTANT: Do NOT resize to original size!
        # Kaggle submission requires fixed 256x256 size for all masks
        # Return the model output directly at 256x256 resolution
        mask_binary = (binary_mask > 0.5).astype(np.uint8)
        
        return mask_binary
    
    def infer_single_image(
        self,
        image_path: str,
        threshold: float = 0.5,
        return_confidence: bool = False,
    ) -> Dict:
        """
        Run inference on single image
        
        Args:
            image_path: Path to input image
            threshold: Threshold for binarization
            return_confidence: Whether to return confidence map
            
        Returns:
            Dictionary with 'mask', 'confidence' (optional), and 'shape'
        """
        with torch.no_grad():
            # Preprocess
            image_tensor, original_size = self.preprocess_image(image_path)
            
            # Forward pass
            output = self.model(image_tensor)
            
            # Postprocess
            mask = self.postprocess_mask(output, original_size, threshold)
            
            result = {
                'mask': mask,
                'shape': mask.shape,
                'image_path': image_path,
            }
            
            if return_confidence:
                # Return confidence map (probabilities before threshold)
                confidence = torch.sigmoid(output).squeeze().cpu().numpy()
                confidence_resized = Image.fromarray((confidence * 255).astype(np.uint8))
                confidence_resized = confidence_resized.resize((original_size[0], original_size[1]))
                result['confidence'] = np.array(confidence_resized).astype(np.float32) / 255.0
            
            return result
    
    def infer_batch(
        self,
        image_paths: List[str],
        threshold: float = 0.5,
    ) -> List[Dict]:
        """
        Run inference on multiple images
        
        Args:
            image_paths: List of paths to input images
            threshold: Threshold for binarization
            
        Returns:
            List of result dictionaries
        """
        results = []
        
        with torch.no_grad():
            pbar = tqdm(image_paths, desc="Inferring")
            for image_path in pbar:
                result = self.infer_single_image(image_path, threshold, return_confidence=False)
                results.append(result)
        
        return results
    
    def infer_directory(
        self,
        image_dir: str,
        output_dir: str = 'inference_output',
        threshold: float = 0.5,
        save_masks: bool = True,
    ) -> List[Dict]:
        """
        Run inference on all images in a directory
        
        Args:
            image_dir: Path to directory with images
            output_dir: Path to save output masks
            threshold: Threshold for binarization
            save_masks: Whether to save output masks
            
        Returns:
            List of result dictionaries
        """
        image_dir = Path(image_dir)
        image_paths = sorted(list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png')))
        
        print(f"Found {len(image_paths)} images in {image_dir}")
        
        # Create output directory
        if save_masks:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
        
        results = []
        
        with torch.no_grad():
            pbar = tqdm(image_paths, desc="Inferring")
            for image_path in pbar:
                result = self.infer_single_image(str(image_path), threshold)
                results.append(result)
                
                # Save mask
                if save_masks:
                    mask_array = (result['mask'] * 255).astype(np.uint8)
                    mask_image = Image.fromarray(mask_array)
                    output_path = output_dir / f"{image_path.stem}_mask.png"
                    mask_image.save(output_path)
        
        print(f"Inference completed! {len(results)} masks generated.")
        
        if save_masks:
            print(f"Output masks saved to {output_dir}")
        
        return results





def main():
    """
    Generate predictions for test set and output Kaggle submission CSV
    """
    parser = argparse.ArgumentParser(
        description="Generate Kaggle submission from test set"
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['unet', 'resnet34_unet'],
        help='Model architecture'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint (.pth file)'
    )
    parser.add_argument(
        '--test_list',
        type=str,
        required=True,
        help='Path to test image list (one image ID per line)'
    )
    parser.add_argument(
        '--image_dir',
        type=str,
        required=True,
        help='Directory containing images'
    )
    parser.add_argument(
        '--output_csv',
        type=str,
        required=True,
        help='Path to output CSV file for Kaggle submission'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Binarization threshold (default: 0.5)'
    )
    parser.add_argument(
        '--image_size',
        type=int,
        default=256,
        help='Target image size for model input (default: 256)'
    )
    
    args = parser.parse_args()
    
    # Read test image IDs
    test_list_path = Path(args.test_list)
    if not test_list_path.exists():
        raise FileNotFoundError(f"Test list not found: {args.test_list}")
    
    with open(test_list_path, 'r') as f:
        test_ids = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(test_ids)} test images from {args.test_list}")
    
    # Get device
    device = get_device()
    
    # Load model
    print(f"Loading model: {args.model}")
    if args.model == 'unet':
        model = UNet(in_channels=3, out_channels=1)
    elif args.model == 'resnet34_unet':
        model = ResNet34UNet(in_channels=3, out_channels=1)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    # Create inference engine
    inference = SegmentationInference(
        model=model,
        checkpoint_path=args.checkpoint,
        device=device,
        image_size=(args.image_size, args.image_size),
    )
    
    # Generate predictions and write to CSV
    print(f"Generating predictions and writing to: {args.output_csv}\n")
    
    image_dir = Path(args.image_dir)
    
    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_id', 'encoded_mask'])
        
        with torch.no_grad():
            for image_id in tqdm(test_ids, desc="Inference & RLE Encoding"):
                # Try common image extensions
                image_path = None
                for ext in ['.jpg', '.jpeg', '.png']:
                    candidate = image_dir / f"{image_id}{ext}"
                    if candidate.exists():
                        image_path = candidate
                        break
                
                if image_path is None:
                    print(f"Warning: Image not found for {image_id}, skipping")
                    continue
                
                # Run inference
                result = inference.infer_single_image(
                    str(image_path),
                    threshold=args.threshold,
                    return_confidence=False,
                )
                mask = result['mask']
                
                # Encode RLE
                rle_string = encode_rle(mask)
                
                # Write to CSV
                writer.writerow([image_id, rle_string])
    
    print(f"\nSubmission saved to {args.output_csv}")
    print(f"Total predictions: {len(test_ids)}")


if __name__ == '__main__':
    main()
