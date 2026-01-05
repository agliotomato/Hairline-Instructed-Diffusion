# Running on Lambda AI

This guide describes how to set up and run the **SD3.5 Dual-Stream ControlNet training** on a Lambda AI instance.

## 1. Environment Setup

After connecting to your instance (e.g., via SSH or Jupyter terminal), clone your repository or upload the files.

Then, update the installed packages to ensure SD3.5 compatibility:

```bash
pip install -r requirements.txt
# Alternatively, install key packages directly to ensure latest versions:
pip install --upgrade diffusers transformers accelerate huggingface_hub
```

**Authentication**:
You need to be authenticated with Hugging Face to download the gated SD3.5 model.
```bash
huggingface-cli login
# Enter your HF Token (Write access not strictly needed, but Read access to stabilityai/stable-diffusion-3.5-medium is required)
```

## 2. Model & Data

Ensure your dataset is present on the instance. Structure should match what `HairlineDatasetV2` expects:
- `data/original_images`
- `data/bald_images`
- `data/masks`

(Optional) If you have a custom metadata file, ensure it is uploaded as well.

## 3. Running Training

Run the training script with the following command. Adjust `train_batch_size` and `resolution` based on your GPU VRAM (SD3.5 is heavy!).

**Recommended for A100 (80GB):**
```bash
accelerate launch train_hairline_cond_sd3.py \
  --pretrained_model_name_or_path "stabilityai/stable-diffusion-3.5-medium" \
  --orig_dir "data/original_images" \
  --bald_dir "data/bald_images" \
  --mask_dir "data/semantic_masks" \
  --output_dir "output/sd3_experiment_1" \
  --resolution 1024 \
  --train_batch_size 2 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-5 \
  --mixed_precision "fp16" \
  --checkpointing_steps 500 \
  --num_train_epochs 20
```

**Low VRAM Mode (Testing):**
If you hit OOM, try:
- `--train_batch_size 1`
- `--resolution 512` (Note: SD3 is optimized for 1024, quality might drop)
- Enable gradient checkpointing (add `--gradient_checkpointing` if script supports it, or modify script to enable it on models)

## 4. Monitoring

Training logs will be saved to `output/sd3_experiment_1/logs`.
You can use TensorBoard (if port forwarding is set up) or check the logs manually.
