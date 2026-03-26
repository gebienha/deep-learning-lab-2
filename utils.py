"""
Utility functions for training, evaluation, and logging
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn


def dice_score(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Calculate Dice score (F1 score for binary segmentation)
    
    Formula: Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    
    Args:
        predictions: Model predictions (B, 1, H, W) with values in [0, 1]
        targets: Ground truth labels (B, 1, H, W) with values in {0, 1}
        threshold: Threshold for binarizing predictions
        
    Returns:
        Dice score value between 0 and 1
    """
    # Binarize predictions
    preds_binary = (predictions >= threshold).float()
    
    # Flatten
    preds_flat = preds_binary.view(-1)
    targets_flat = targets.view(-1)
    
    # Calculate intersection and union
    intersection = torch.sum(preds_flat * targets_flat)
    union = torch.sum(preds_flat) + torch.sum(targets_flat)
    
    # Calculate Dice
    dice = 2.0 * intersection / (union + 1e-8)
    
    return dice.item()


def iou_score(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Calculate Intersection over Union (IoU) score
    
    Formula: IoU = |X ∩ Y| / |X ∪ Y|
    
    Args:
        predictions: Model predictions (B, 1, H, W) with values in [0, 1]
        targets: Ground truth labels (B, 1, H, W) with values in {0, 1}
        threshold: Threshold for binarizing predictions
        
    Returns:
        IoU score value between 0 and 1
    """
    # Binarize predictions
    preds_binary = (predictions >= threshold).float()
    
    # Flatten
    preds_flat = preds_binary.view(-1)
    targets_flat = targets.view(-1)
    
    # Calculate intersection and union
    intersection = torch.sum(preds_flat * targets_flat)
    union = torch.sum(preds_flat) + torch.sum(targets_flat) - intersection
    
    # Calculate IoU
    iou = intersection / (union + 1e-8)
    
    return iou.item()


def save_checkpoint(
    model: nn.Module,
    optimizer,
    epoch: int,
    metrics: Dict,
    save_path: str,
) -> None:
    """
    Save model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        metrics: Dictionary of metrics
        save_path: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(
    model: nn.Module,
    optimizer,
    checkpoint_path: str,
    device: str = 'cpu',
) -> Tuple[int, Dict]:
    """
    Load model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        checkpoint_path: Path to checkpoint
        device: Device to load on
        
    Returns:
        Tuple of (epoch, metrics)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    metrics = checkpoint['metrics']
    
    print(f"Checkpoint loaded from {checkpoint_path}")
    
    return epoch, metrics


def save_metrics(metrics_history: Dict[str, List], save_path: str) -> None:
    """
    Save metrics history to JSON
    
    Args:
        metrics_history: Dictionary with metric histories
        save_path: Path to save metrics
    """
    # Convert numpy arrays to lists for JSON serialization
    metrics_json = {}
    for key, values in metrics_history.items():
        metrics_json[key] = [float(v) if isinstance(v, (np.floating, torch.Tensor)) else v for v in values]
    
    with open(save_path, 'w') as f:
        json.dump(metrics_json, f, indent=2)
    
    print(f"Metrics saved to {save_path}")


def load_metrics(metrics_path: str) -> Dict[str, List]:
    """
    Load metrics history from JSON
    
    Args:
        metrics_path: Path to metrics file
        
    Returns:
        Dictionary with metric histories
    """
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    return metrics


def get_device() -> str:
    """
    Get device (GPU if available, else CPU)
    
    Returns:
        Device string ('cuda' or 'cpu')
    """
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("Using CPU")
    
    return device


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters in model
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


def print_model_summary(model: nn.Module) -> None:
    """
    Print model summary
    
    Args:
        model: PyTorch model
    """
    total_params, trainable_params = count_parameters(model)
    
    print("=" * 60)
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("=" * 60)


def create_save_dirs(model_name: str = 'unet') -> str:
    """
    Create directories for saving models and metrics
    
    Args:
        model_name: Name of model
        
    Returns:
        Path to saved models directory
    """
    save_dir = Path("saved_models")
    save_dir.mkdir(exist_ok=True)
    
    return str(save_dir)


if __name__ == '__main__':
    # Test utility functions
    device = get_device()
    
    # Test dice score
    preds = torch.tensor([[[[0.9, 0.1], [0.8, 0.2]]]])
    targets = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    dice = dice_score(preds, targets)
    print(f"Dice score: {dice:.4f}")
    
    # Test IoU score
    iou = iou_score(preds, targets)
    print(f"IoU score: {iou:.4f}")
