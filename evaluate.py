"""
Evaluation script for semantic segmentation models

Usage:
    python src/evaluate.py --model unet --checkpoint saved_models/unet_best.pth
    python src/evaluate.py --model resnet34_unet --checkpoint saved_models/resnet34_unet_best.pth
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.unet import UNet
from models.resnet34_unet import ResNet34UNet
from oxford_pet import DataPreprocessor
from utils import dice_score, iou_score, get_device


class SegmentationEvaluator:
    """
    Evaluator for semantic segmentation
    """
    
    def __init__(self, model: nn.Module, device: str):
        """
        Initialize evaluator
        
        Args:
            model: PyTorch model
            device: Device to evaluate on
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss()
    
    def evaluate(self, test_loader: DataLoader) -> dict:
        """
        Evaluate model on test set
        
        Args:
            test_loader: Test dataloader
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_dice = 0.0
        total_iou = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc="Evaluating")
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                # Metrics
                pred_sigmoid = torch.sigmoid(outputs)
                dice = dice_score(pred_sigmoid, masks)
                iou = iou_score(pred_sigmoid, masks)
                
                total_loss += loss.item()
                total_dice += dice
                total_iou += iou
                num_batches += 1
                
                pbar.set_postfix({
                    'loss': total_loss / num_batches,
                    'dice': total_dice / num_batches,
                    'iou': total_iou / num_batches,
                })
        
        return {
            'loss': total_loss / num_batches,
            'dice': total_dice / num_batches,
            'iou': total_iou / num_batches,
        }


def main():
    """
    Main evaluation function
    """
    parser = argparse.ArgumentParser(description="Evaluate semantic segmentation model")
    parser.add_argument(
        '--model',
        type=str,
        default='unet',
        choices=['unet', 'resnet34_unet'],
        help='Model architecture'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--image_size', type=int, default=256, help='Image size')
    parser.add_argument('--dataset_dir', type=str, default='dataset/oxford-iiit-pet')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Get device
    device = get_device()
    
    # Load model
    print(f"Loading model: {args.model}...")
    if args.model == 'unet':
        model = UNet(in_channels=3, out_channels=1)
    elif args.model == 'resnet34_unet':
        model = ResNet34UNet(in_channels=3, out_channels=1)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Checkpoint loaded!")
    
    # Load dataset
    print(f"\nLoading dataset from {args.dataset_dir}...")
    preprocessor = DataPreprocessor(
        dataset_dir=args.dataset_dir,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    
    _, _, test_loader = preprocessor.get_loaders()
    print(f"Test set: {len(test_loader.dataset)} images")
    
    # Evaluate
    evaluator = SegmentationEvaluator(model, device)
    print(f"\nEvaluating on test set...")
    metrics = evaluator.evaluate(test_loader)
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Loss:       {metrics['loss']:.4f}")
    print(f"Dice Score: {metrics['dice']:.4f}")
    print(f"IoU Score:  {metrics['iou']:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
