# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

K-LoRA is the official implementation of the CVPR 2025 paper "K-LoRA: Unlocking Training-Free Fusion of Any Subject and Style LoRAs". It enables combining two LoRA models (content + style) to generate images without additional training.

## Common Commands

### Training a LoRA
```bash
accelerate launch train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --instance_data_dir="path/to/images" \
  --output_dir="lora-output" \
  --instance_prompt="a sbu dog" \
  --rank=8 \
  --resolution=1024 \
  --train_batch_size=1 \
  --learning_rate=5e-5 \
  --max_train_steps=1000 \
  --use_8bit_adam \
  --gradient_checkpointing \
  --enable_xformers_memory_efficient_attention
```

### Inference (SDXL)
```bash
python inference_sd.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --lora_name_or_path_content="path/to/content.safetensors" \
  --lora_name_or_path_style="path/to/style.safetensors" \
  --output_folder="output" \
  --prompt="a sbu dog in szn style" \
  --pattern="s*"
```

### Inference (FLUX)
```bash
python inference_flux.py \
  --pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev" \
  --content_index=0 \
  --style_index=1 \
  --output_folder="output" \
  --pattern="s*"
```

### Interactive UI (Gradio)
```bash
python inference_gradio.py \
  --pretrained_model_name_or_path="path/to/model" \
  --lora_name_or_path_content="path/to/content.safetensors" \
  --lora_name_or_path_style="path/to/style.safetensors"
```

## Architecture

### Core Files

- **klora.py**: Defines `KLoRALinearLayer` (training) and `KLoRALinearLayerInference` - custom layers that dynamically merge two LoRA weight matrices based on diffusion timestep
- **utils.py**: LoRA loading (`get_lora_weights`), weight merging (`merge_lora_weights`, `merge_community_flux_lora_weights`), and UNet injection functions (`insert_sd_klora_to_unet`, `insert_community_flux_lora_to_unet`)
- **train_dreambooth_lora_sdxl.py**: Full DreamBooth LoRA training pipeline using HuggingFace Diffusers and Accelerate

### K-LoRA Merging Mechanism

The core innovation is timestep-aware weight selection:
1. Both content and style LoRA weights are loaded into `KLoRALinearLayer`
2. During inference, a time-dependent scale is computed: `scale = alpha * time_ratio / sum_timesteps + beta`
3. For pattern `"s*"` (recommended): `scale = scale % alpha` creates cyclic blending
4. Top-k values from both matrices are compared; the larger magnitude determines which weight is used
5. `average_ratio` normalizes content-to-style magnitude differences

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | 1.5 | Scaling factor for blending curve |
| `beta` | 1.275 | Baseline offset (1.5 * 0.85 for s*) |
| `sum_timesteps` | 28000 | Total diffusion steps tracked |
| `pattern` | "s*" | "s" (linear) or "s*" (modular, recommended) |

### Supported Models

- **SDXL**: Primary target, uses `insert_sd_klora_to_unet`
- **FLUX**: Community LoRA support via `insert_community_flux_lora_to_unet`, handles 190-layer and 494-layer variants

### LoRA Attention Targets

Weights are injected into: `to_q`, `to_k`, `to_v`, `to_out[0]` (attention projections)

## Prompt Conventions

- Content trigger: `sbu` (e.g., "a sbu dog")
- Style trigger: `szn` (e.g., "in szn style")
- Combined: "a sbu dog in szn style"
