# Lab 2: Binary Semantic Segmentation Workflow

## Three-Step Pipeline

### Step 1: Training
Train the model and save the best checkpoint.

```bash
./venv/bin/python src/train.py \
  --model unet \
  --epochs 5 \
  --batch_size 8
```

**Output:**
- `saved_models/unet_best.pth` - Best model checkpoint
- `saved_models/unet_metrics.json` - Training/validation metrics
- `saved_models/dataset_split.json` - Train/val/test split info

**Available models:** `unet` or `resnet34_unet`

---

### Step 2: Inference (Generate PNG Masks)
Generate PNG mask predictions from the checkpoint using the images.

```bash
./venv/bin/python src/inference.py \
  --model unet \
  --checkpoint saved_models/unet_best.pth \
  --image_dir dataset/oxford-iiit-pet/images \
  --output_dir inference_output/ \
  --threshold 0.5
```

**Output:**
- `inference_output/image_1_mask.png`
- `inference_output/image_2_mask.png`
- ... (one PNG per image, binary: 0 or 255)

**Parameters:**
- `--threshold`: Binarization threshold (default: 0.5)
- PNG masks are 8-bit: background=0, foreground=255

---

### Step 3: Convert to Kaggle CSV (RLE Encoding)
Convert PNG masks to Kaggle submission CSV with Run-Length Encoding.

```bash
./venv/bin/python src/masks_to_csv.py \
  --mask_dir inference_output/ \
  --output_csv submission.csv
```

**Output:**
- `submission.csv` - Kaggle submission format
  ```
  image_id,encoded_mask
  image_1,RLE_string_1
  image_2,RLE_string_2
  ...
  ```

**Format Details:**
- `image_id`: Image filename without extension
- `encoded_mask`: Run-Length Encoding in column-major (Fortran) order
- Binary masks: 0 (background), 1 (foreground)
- Encoded as space-separated run lengths

---

## Example Complete Workflow

```bash
# Step 1: Train for 50 epochs
./venv/bin/python src/train.py --model unet --epochs 50 --batch_size 8

# Step 2: Generate predictions
./venv/bin/python src/inference.py \
  --model unet \
  --checkpoint saved_models/unet_best.pth \
  --image_dir dataset/oxford-iiit-pet/images \
  --output_dir inference_output/

# Step 3: Create Kaggle submission
./venv/bin/python src/masks_to_csv.py \
  --mask_dir inference_output/ \
  --output_csv submission.csv
```

---

## Testing on Single Image

For quick testing, use `evaluate.py`:

```bash
./venv/bin/python src/evaluate.py \
  --model unet \
  --checkpoint saved_models/unet_best.pth
```

This evaluates on the test set and prints Dice/IoU scores.

---

## Key Features

### Separation of Concerns
- **train.py**: Training only - saves `.pth` checkpoints
- **inference.py**: Prediction only - loads checkpoint, outputs PNG masks
- **masks_to_csv.py**: Conversion only - PNG → Kaggle CSV with RLE

### Dataset
- **Images:** 256×256 RGB, normalized with ImageNet stats
- **Masks:** 256×256 binary {0, 1}
- **Trimap conversion:** 1 → 1.0 (foreground), 2,3 → 0.0 (background)

### Architecture Options
1. **UNet** (31.4M parameters)
   - 4 encoder blocks + 4 decoder blocks
   - Skip connections
   - Bilinear upsampling

2. **ResNet34-UNet** (25.2M parameters)
   - ResNet34 encoder
   - 4 decoder blocks
   - Skip connections from encoder features

### Training Parameters
- **Loss:** BCEWithLogitsLoss (binary cross-entropy)
- **Optimizer:** Adam (lr=1e-3, weight_decay=1e-5)
- **Scheduler:** ReduceLROnPlateau (factor=0.5, patience=5)
- **Early stopping:** patience=10 epochs
- **Metrics:** Dice Score (primary), IoU Score (secondary)

---

## Troubleshooting

### "No images found"
Check that `--image_dir` contains PNG/JPG files

### "Checkpoint not found"
Verify checkpoint path is correct, e.g., `saved_models/unet_best.pth`

### "Invalid mask directory"
Ensure masks have correct suffix `_mask.png`

---

## Files

- `src/train.py` - Training pipeline (Step 1)
- `src/inference.py` - PNG mask generation (Step 2)
- `src/masks_to_csv.py` - RLE encoding (Step 3)
- `src/evaluate.py` - Test set evaluation
- `src/oxford_pet.py` - Dataset loading & preprocessing
- `src/utils.py` - Utility functions
- `src/models/unet.py` - UNet architecture
- `src/models/resnet34_unet.py` - ResNet34-UNet architecture
