#!/bin/bash

# Run multiple LoRA combinations for comparison
# Each combination generates 5 samples with comparison mode (5 images per sample = 25 images per combo)

# Combinations: content_index, style_index, lighting_index
# 0,0,0 - already done
# 1,1,1
# 2,2,0
# 3,3,1

echo "=== Running LoRA Combinations ==="

# Combination 1,1,1: marv + acrylic + cyberpunk
echo ""
echo ">>> Combination 1,1,1: marv + acrylic + cyberpunk"
python inference_flux.py \
    --content_index=1 --style_index=1 --lighting_index=1 \
    --comparison_mode --num_samples=5 \
    --output_folder=output_combo_1_1_1

# Combination 2,2,0: cats + industrial + cinematic
echo ""
echo ">>> Combination 2,2,0: cats + industrial + cinematic"
python inference_flux.py \
    --content_index=2 --style_index=2 --lighting_index=0 \
    --comparison_mode --num_samples=5 \
    --output_folder=output_combo_2_2_0

# Combination 3,3,1: chillguy + bluedraw + cyberpunk
echo ""
echo ">>> Combination 3,3,1: chillguy + bluedraw + cyberpunk"
python inference_flux.py \
    --content_index=3 --style_index=3 --lighting_index=1 \
    --comparison_mode --num_samples=5 \
    --output_folder=output_combo_3_3_1

echo ""
echo "=== All combinations complete ==="
echo "Output folders:"
echo "  - output_combo_1_1_1/"
echo "  - output_combo_2_2_0/"
echo "  - output_combo_3_3_1/"
