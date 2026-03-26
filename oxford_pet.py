"""
Oxford-IIIT Pet Dataset Loader and Preprocessing

Handles dataset loading, preprocessing, and binary mask conversion
"""

import os
import json
import tarfile
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class OxfordPetDataset(Dataset):
    """
    Oxford-IIIT Pet Dataset for semantic segmentation
    
    Attributes:
        image_dir: Path to images directory
        mask_dir: Path to segmentation masks (trimaps) directory
        image_ids: List of image IDs
        image_size: Target image size (H, W)
        normalize: Whether to normalize images
    """
    
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        image_ids: List[str],
        image_size: Tuple[int, int] = (256, 256),
        normalize: bool = True,
        augmentation: bool = False,
    ):
        """
        Initialize dataset
        
        Args:
            image_dir: Path to directory containing images
            mask_dir: Path to directory containing trimaps
            image_ids: List of image file names (without extension)
            image_size: Target size for resizing
            normalize: Whether to normalize images
            augmentation: Whether to apply data augmentation
        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_ids = image_ids
        self.image_size = image_size
        self.normalize = normalize
        self.augmentation = augmentation
        
        # Image normalization parameters (ImageNet mean/std)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        # Augmentation transforms
        self.aug_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
        ]) if augmentation else None
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get single sample
        
        Args:
            idx: Index in dataset
            
        Returns:
            Tuple of (image tensor, binary mask tensor)
        """
        # Load image
        image_path = self.image_dir / f"{self.image_ids[idx]}.jpg"
        image = Image.open(image_path).convert('RGB')
        
        # Load trimap mask
        mask_path = self.mask_dir / f"{self.image_ids[idx]}.png"
        mask = Image.open(mask_path)
        
        # Convert trimap to binary: 1->1, 2->0, 3->0
        mask_array = np.array(mask)
        binary_mask = (mask_array == 1).astype(np.float32)
        
        # Resize
        image = image.resize(self.image_size, Image.Resampling.BILINEAR)
        mask_pil = Image.fromarray((binary_mask * 255).astype(np.uint8))
        mask_pil = mask_pil.resize(self.image_size, Image.Resampling.NEAREST)
        binary_mask = np.array(mask_pil).astype(np.float32) / 255.0
        
        # Apply augmentation (on both image and mask)
        if self.aug_transforms is not None:
            # For augmentation, temporarily convert back to PIL
            seed = torch.seed()
            image_pil = image
            image_pil = self.aug_transforms(image_pil)
            
            # Apply same augmentation to mask
            torch.manual_seed(seed)
            mask_pil = Image.fromarray((binary_mask * 255).astype(np.uint8))
            mask_pil = self.aug_transforms(mask_pil)
            binary_mask = np.array(mask_pil).astype(np.float32) / 255.0
            
            image = image_pil
        
        # Convert image to tensor and normalize
        image_tensor = transforms.ToTensor()(image)  # (3, H, W)
        if self.normalize:
            image_tensor = (image_tensor - self.mean) / self.std
        
        # Convert mask to tensor
        mask_tensor = torch.from_numpy(binary_mask).unsqueeze(0)  # (1, H, W)
        
        return image_tensor, mask_tensor


class DatasetSplitter:
    """
    Splits Oxford-IIIT Pet dataset into train/val/test sets
    """
    
    def __init__(
        self,
        dataset_dir: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ):
        """
        Initialize dataset splitter
        
        Args:
            dataset_dir: Path to oxford-iiit-pet directory
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            seed: Random seed for reproducibility
        """
        self.dataset_dir = Path(dataset_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
    
    def split(self) -> Dict[str, List[str]]:
        """
        Split dataset into train/val/test
        
        Returns:
            Dictionary with 'train', 'val', 'test' keys containing image IDs
        """
        # Get all image files
        image_dir = self.dataset_dir / "images"
        all_images = sorted([f.stem for f in image_dir.glob("*.jpg")])
        
        # Set random seed
        np.random.seed(self.seed)
        
        # Shuffle
        indices = np.arange(len(all_images))
        np.random.shuffle(indices)
        shuffled_images = [all_images[i] for i in indices]
        
        # Split
        n_total = len(shuffled_images)
        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)
        
        train_ids = shuffled_images[:n_train]
        val_ids = shuffled_images[n_train:n_train + n_val]
        test_ids = shuffled_images[n_train + n_val:]
        
        return {
            'train': train_ids,
            'val': val_ids,
            'test': test_ids,
        }


class DataPreprocessor:
    """
    Handles dataset loading and preprocessing
    """
    
    def __init__(
        self,
        dataset_dir: str,
        image_size: Tuple[int, int] = (256, 256),
        batch_size: int = 8,
        num_workers: int = 0,
        seed: int = 42,
        train_list: str = None,
        val_list: str = None,
    ):
        """
        Initialize preprocessor
        
        Args:
            dataset_dir: Path to oxford-iiit-pet directory
            image_size: Target image size
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            seed: Random seed
            train_list: Path to file with training image IDs (optional)
            val_list: Path to file with validation image IDs (optional)
        """
        self.dataset_dir = Path(dataset_dir)
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        
        # Paths
        self.image_dir = self.dataset_dir / "images"
        self.mask_dir = self.dataset_dir / "annotations" / "trimaps"
        
        # Load train/val splits from files or use automatic split
        if train_list and val_list:
            self.splits = self._load_splits_from_files(train_list, val_list)
        else:
            # Fallback to automatic split
            splitter = DatasetSplitter(
                dataset_dir=dataset_dir,
                seed=seed,
            )
            self.splits = splitter.split()
    
    def _load_splits_from_files(self, train_list: str, val_list: str) -> Dict[str, List[str]]:
        """
        Load train and val splits from text files
        
        Args:
            train_list: Path to file with training image IDs (one per line)
            val_list: Path to file with validation image IDs (one per line)
            
        Returns:
            Dictionary with 'train' and 'val' keys
        """
        with open(train_list, 'r') as f:
            train_ids = [line.strip() for line in f if line.strip()]
        
        with open(val_list, 'r') as f:
            val_ids = [line.strip() for line in f if line.strip()]
        
        print(f"Loaded {len(train_ids)} training images from {train_list}")
        print(f"Loaded {len(val_ids)} validation images from {val_list}")
        
        return {
            'train': train_ids,
            'val': val_ids,
            'test': [],  # No test split when using external lists
        }
    
    def get_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Get train, val, test dataloaders
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Training dataset (with augmentation)
        train_dataset = OxfordPetDataset(
            image_dir=str(self.image_dir),
            mask_dir=str(self.mask_dir),
            image_ids=self.splits['train'],
            image_size=self.image_size,
            normalize=True,
            augmentation=True,
        )
        
        # Validation dataset (no augmentation)
        val_dataset = OxfordPetDataset(
            image_dir=str(self.image_dir),
            mask_dir=str(self.mask_dir),
            image_ids=self.splits['val'],
            image_size=self.image_size,
            normalize=True,
            augmentation=False,
        )
        
        # Test dataset (no augmentation)
        test_dataset = OxfordPetDataset(
            image_dir=str(self.image_dir),
            mask_dir=str(self.mask_dir),
            image_ids=self.splits['test'],
            image_size=self.image_size,
            normalize=True,
            augmentation=False,
        )
        
        # Dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        
        return train_loader, val_loader, test_loader
    
    def get_split_ids(self) -> Dict[str, List[str]]:
        """
        Get the image IDs for each split
        
        Returns:
            Dictionary with 'train', 'val', 'test' keys
        """
        return self.splits


if __name__ == '__main__':
    # Test dataset loading
    dataset_dir = "dataset/oxford-iiit-pet"
    
    if Path(dataset_dir).exists():
        print(f"Loading dataset from {dataset_dir}...")
        
        # Get preprocessor
        preprocessor = DataPreprocessor(
            dataset_dir=dataset_dir,
            image_size=(256, 256),
            batch_size=4,
            num_workers=0,
        )
        
        # Get loaders
        train_loader, val_loader, test_loader = preprocessor.get_loaders()
        
        print(f"Train set size: {len(train_loader.dataset)}")
        print(f"Val set size: {len(val_loader.dataset)}")
        print(f"Test set size: {len(test_loader.dataset)}")
        
        # Test batch
        images, masks = next(iter(train_loader))
        print(f"\nBatch shapes:")
        print(f"Images: {images.shape}")
        print(f"Masks: {masks.shape}")
        print(f"Image range: [{images.min():.4f}, {images.max():.4f}]")
        print(f"Mask range: [{masks.min():.4f}, {masks.max():.4f}]")
    else:
        print(f"Dataset not found at {dataset_dir}")
