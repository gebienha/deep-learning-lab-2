"""
Training script for binary semantic segmentation on Oxford-IIIT Pet Dataset

Usage:
    python src/train.py --model unet --epochs 50 --batch_size 8
    python src/train.py --model resnet34_unet --epochs 50 --batch_size 8
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.unet import UNet
from models.resnet34_unet import ResNet34UNet
from oxford_pet import DataPreprocessor
from utils import (
    dice_score, iou_score, save_checkpoint, save_metrics,
    get_device, count_parameters, print_model_summary, create_save_dirs
)


class SegmentationTrainer:
    """
    Trainer for semantic segmentation
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        """
        Initialize trainer
        
        Args:
            model: PyTorch model
            device: Device to train on
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
        """
        self.model = model.to(device)
        self.device = device
        
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
        )
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            train_loader: Training dataloader
            
        Returns:
            Dictionary with epoch metrics
        """
        self.model.train()
        
        total_loss = 0.0
        total_dice = 0.0
        total_iou = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            with torch.no_grad():
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
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate on validation set
        
        Args:
            val_loader: Validation dataloader
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_dice = 0.0
        total_iou = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validating")
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
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        model_name: str = 'unet',
        save_dir: str = 'saved_models',
        early_stopping_patience: int = 10,
    ) -> Dict:
        """
        Complete training loop
        
        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader
            num_epochs: Number of epochs to train
            model_name: Name of model for saving
            save_dir: Directory to save checkpoints
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Dictionary with training history
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        history = {
            'train_loss': [],
            'train_dice': [],
            'train_iou': [],
            'val_loss': [],
            'val_dice': [],
            'val_iou': [],
        }
        
        best_val_loss = float('inf')
        epochs_no_improve = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 60)
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            history['train_loss'].append(train_metrics['loss'])
            history['train_dice'].append(train_metrics['dice'])
            history['train_iou'].append(train_metrics['iou'])
            
            print(f"Train Loss: {train_metrics['loss']:.4f} | "
                  f"Dice: {train_metrics['dice']:.4f} | "
                  f"IoU: {train_metrics['iou']:.4f}")
            
            # Validate
            val_metrics = self.validate(val_loader)
            history['val_loss'].append(val_metrics['loss'])
            history['val_dice'].append(val_metrics['dice'])
            history['val_iou'].append(val_metrics['iou'])
            
            print(f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"Dice: {val_metrics['dice']:.4f} | "
                  f"IoU: {val_metrics['iou']:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['loss'])
            
            # Save best checkpoint
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                epochs_no_improve = 0
                
                checkpoint_path = save_dir / f"{model_name}_best.pth"
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_metrics,
                    str(checkpoint_path),
                )
            else:
                epochs_no_improve += 1
            
            # Early stopping
            if epochs_no_improve >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break
        
        # Save metrics
        metrics_path = save_dir / f"{model_name}_metrics.json"
        save_metrics(history, str(metrics_path))
        
        return history


def main():
    """
    Main training function
    """
    parser = argparse.ArgumentParser(description="Train semantic segmentation model")
    parser.add_argument(
        '--model',
        type=str,
        default='unet',
        choices=['unet', 'resnet34_unet'],
        help='Model architecture to train'
    )
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--image_size', type=int, default=256, help='Image size')
    parser.add_argument('--dataset_dir', type=str, default='dataset/oxford-iiit-pet')
    parser.add_argument('--save_dir', type=str, default='saved_models')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--train_list', type=str, default='train.txt', help='Path to train.txt')
    parser.add_argument('--val_list', type=str, default='val.txt', help='Path to val.txt')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Get device
    device = get_device()
    
    # Create save directory
    save_dir = create_save_dirs(args.model)
    
    # Load dataset
    print(f"\nLoading dataset from {args.dataset_dir}...")
    preprocessor = DataPreprocessor(
        dataset_dir=args.dataset_dir,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        train_list=args.train_list,
        val_list=args.val_list,
    )
    
    train_loader, val_loader, test_loader = preprocessor.get_loaders()
    split_ids = preprocessor.get_split_ids()
    
    print(f"Train set: {len(train_loader.dataset)} images")
    print(f"Val set: {len(val_loader.dataset)} images")
    print(f"Test set: {len(test_loader.dataset)} images")
    
    # Save split IDs
    split_path = Path(save_dir) / "dataset_split.json"
    with open(split_path, 'w') as f:
        json.dump(split_ids, f, indent=2)
    print(f"Dataset split saved to {split_path}")
    
    # Create model
    print(f"\nCreating model: {args.model}...")
    if args.model == 'unet':
        model = UNet(in_channels=3, out_channels=1)
    elif args.model == 'resnet34_unet':
        model = ResNet34UNet(in_channels=3, out_channels=1)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    print_model_summary(model)
    
    # Create trainer
    trainer = SegmentationTrainer(
        model=model,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    # Train
    print(f"\nTraining for {args.epochs} epochs...\n")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        model_name=args.model,
        save_dir=save_dir,
    )
    
    print("\nTraining completed!")
    print(f"Best model saved to: saved_models/{args.model}_best.pth")
    print(f"Metrics saved to: saved_models/{args.model}_metrics.json")


if __name__ == '__main__':
    main()
