1. install requirements
2. login hf  huggingface-cli login
3. python inference_flux.py \
    --pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev" \
    --content_index=0 \
    --style_index=1 \
    --output_folder="/root/K-LoRA/output" \
    --num_samples=10 \
    --pattern="s*"