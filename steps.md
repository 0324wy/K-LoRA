1. install requirements
2. login hf  huggingface-cli login
3. python inference_flux.py \
    --pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev" \
    --content_index=0 \
    --style_index=1 \
    --output_folder="/root/K-LoRA/output" \
    --num_samples=10 \
    --pattern="s*"

 4.     # Normal 3-LoRA mode
  python inference_flux.py \
    --content_index=0 --style_index=0 --lighting_index=0 \
    --gamma=0.3 --output_folder=output_3lora

 5. # Comparison mode (5 images per seed)
  python inference_flux.py \
    --content_index=0 --style_index=0 --lighting_index=0 \
    --comparison_mode --num_samples=1 --output_folder=output_comparison