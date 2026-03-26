# Quick Start Guide

## Your Three-Step Workflow

### 1️⃣ Train Model (currently running: 5 epochs test)
```bash
./venv/bin/python src/train.py --model unet --epochs 50 --batch_size 8
```
Creates: `saved_models/unet_best.pth`

### 2️⃣ Generate PNG Masks (after training)
```bash
./venv/bin/python src/inference.py \
  --model unet \
  --checkpoint saved_models/unet_best.pth \
  --image_dir dataset/oxford-iiit-pet/images \
  --output_dir inference_output/
```
Creates: PNG files in `inference_output/`

### 3️⃣ Convert to Kaggle CSV (after masks)
```bash
./venv/bin/python src/masks_to_csv.py \
  --mask_dir inference_output/ \
  --output_csv submission.csv
```
Creates: `submission.csv` with RLE encoding

---

## CSV Format Explanation

Your Kaggle submission should look like:

```
image_id,encoded_mask
image_1,2 2 2 2 ...
image_2,1 3 1 2 ...
...
```

- **image_id**: Filename without extension (e.g., "image_1" from "image_1.jpg")
- **encoded_mask**: Run-length encoding in column-major (Fortran) order
  - Alternates between run lengths of 0s and 1s
  - Space-separated numbers
  - Example: "2 2 3 1" = [0,0,1,1,1,0]

---

## Files Created

✅ **Training**
- `src/train.py` - Full pipeline with early stopping

✅ **Inference**  
- `src/inference.py` - Generate PNG masks from checkpoint

✅ **Conversion**
- `src/masks_to_csv.py` - PNG to RLE CSV encoding

✅ **Models**
- `src/models/unet.py` - UNet (31.4M params)
- `src/models/resnet34_unet.py` - ResNet34-UNet (25.2M params)

✅ **Utils**
- `src/oxford_pet.py` - Dataset & preprocessing
- `src/utils.py` - Metrics, checkpoints, helpers
- `src/evaluate.py` - Test set evaluation

---

## Current Status

🟡 **Training Status**: Epoch 1/5 in progress
- Next: Wait for 5 epochs to complete (~2-3 hours on CPU)
- Then: Run step 2 (inference)
- Then: Run step 3 (CSV conversion)

---

## Key Parameters

**Training**
- Loss: BCEWithLogitsLoss
- Optimizer: Adam (lr=1e-3)
- Batch size: 8
- Image size: 256×256
- Augmentation: H/V flip, rotation

**Inference**
- Threshold: 0.5 (adjustable)
- Output: PNG (binary 0/255)

**Metrics**
- Dice Score (primary)
- IoU Score (secondary)
- Threshold: 0.5

---

## Next Steps

1. ✅ Training running (5 epochs test)
2. ⏳ Wait for training to complete
3. ⏳ Run inference to generate PNG masks
4. ⏳ Run masks_to_csv to create submission

Monitor training progress by checking the terminal output.
