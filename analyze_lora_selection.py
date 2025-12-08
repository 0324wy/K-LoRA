"""
Analyze K-LoRA selection patterns: which LoRA is selected per layer per timestep.
Outputs statistics without generating images.
"""

import argparse
import torch
from collections import defaultdict
from utils import get_lora_weights
import json

# LoRA configurations (same as inference_flux.py)
record_content_loras = [
    "ginipick/flux-lora-eric-cat",
    "glif-loradex-trainer/antix82_flux_dev_marv_simplecap_v1",
    "glif-loradex-trainer/festerbitcoin_86601_cats",
    "glif-loradex-trainer/fabian3000_chillguy",
]
content_lora_weight_names = [
    "flux-lora-eric-cat.safetensors",
    "flux_dev_marv_simplecap_v1.safetensors",
    "cats.safetensors",
    "chillguy.safetensors",
]

record_style_loras = [
    "glif-loradex-trainer/bingbangboom_flux_surf",
    "glif-loradex-trainer/mindlywork_AcrylicWorld",
    "glif-loradex-trainer/an303042_Seiwert_Industrial_v1",
    "glif-loradex-trainer/maxxd4240_BlueDraw",
]
style_lora_weight_names = [
    "flux_surf.safetensors",
    "AcrylicWorld.safetensors",
    "Seiwert_Industrial_v1.safetensors",
    "BlueDraw.safetensors",
]

record_lighting_loras = [
    "aixonlab/FLUX.1-dev-LoRA-Cinematic-Octane",
    "fofr/flux-80s-cyberpunk",
]
lighting_lora_weight_names = [
    "cinematic-octane.safetensors",
    "lora.safetensors",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze K-LoRA selection patterns")
    parser.add_argument("--content_index", type=int, default=0)
    parser.add_argument("--style_index", type=int, default=0)
    parser.add_argument("--lighting_index", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--beta", type=float, default=1.275)  # alpha * 0.85
    parser.add_argument("--gamma", type=float, default=0.3)
    parser.add_argument("--num_steps", type=int, default=28, help="Number of diffusion steps")
    parser.add_argument("--pattern", type=str, default="s*")
    parser.add_argument("--output_json", type=str, default=None, help="Save results to JSON file")
    return parser.parse_args()


def get_layer_pairs(lora_weights, device="cuda"):
    """Extract layer name pairs (lora_A, lora_B) from LoRA weights.

    Note: In K-LoRA code, weight_1_a = lora_B and weight_1_b = lora_A (swapped).
    So for matrix product weight_1_a @ weight_1_b, we need B @ A.
    """
    layers = {}
    for key in lora_weights.keys():
        if "lora_A" in key:
            base_name = key.replace(".lora_A.weight", "")
            b_key = key.replace("lora_A", "lora_B")
            if b_key in lora_weights:
                # Swap: "A" stores lora_B, "B" stores lora_A (to match K-LoRA convention)
                layers[base_name] = {
                    "A": lora_weights[b_key].to(device),  # lora_B -> weight_a
                    "B": lora_weights[key].to(device)     # lora_A -> weight_b
                }
    return layers


def simulate_selection(content_layers, style_layers, lighting_layers, args):
    """Simulate K-LoRA selection for each layer at each timestep."""

    num_steps = args.num_steps
    sum_timesteps = num_steps * 1000  # Total timestep budget
    alpha = args.alpha
    beta = args.beta
    gamma = args.gamma
    pattern = args.pattern

    # Get common layer names
    layer_names = list(content_layers.keys())
    num_layers = len(layer_names)

    print(f"\n{'='*60}")
    print(f"K-LoRA Selection Analysis")
    print(f"{'='*60}")
    print(f"Total layers: {num_layers}")
    print(f"Diffusion steps: {num_steps}")
    print(f"Parameters: alpha={alpha}, beta={beta}, gamma={gamma}, pattern={pattern}")
    print(f"{'='*60}")

    # OPTIMIZATION: Precompute top-k sums for all layers (they don't change per timestep)
    print("Precomputing layer magnitudes...")
    layer_topk_sums = []  # List of (topk_content, topk_style, topk_lighting) per layer

    for layer_idx, layer_name in enumerate(layer_names):
        # Get weights for this layer
        content_A = content_layers[layer_name]["A"]
        content_B = content_layers[layer_name]["B"]
        style_A = style_layers[layer_name]["A"]
        style_B = style_layers[layer_name]["B"]
        lighting_A = lighting_layers[layer_name]["A"]
        lighting_B = lighting_layers[layer_name]["B"]

        # Compute matrix products
        matrix1 = content_A @ content_B
        matrix2 = style_A @ style_B
        matrix3 = lighting_A @ lighting_B

        # Compute Top-K sums (use simple sum of abs for speed)
        top_k_sum1 = torch.abs(matrix1).sum().item()
        top_k_sum2 = torch.abs(matrix2).sum().item()
        top_k_sum3 = torch.abs(matrix3).sum().item()

        layer_topk_sums.append((top_k_sum1, top_k_sum2, top_k_sum3))

        if (layer_idx + 1) % 100 == 0:
            print(f"  Processed {layer_idx + 1}/{num_layers} layers...")

    print(f"Done! Now simulating {num_steps} timesteps...\n")

    # Track selections per timestep
    timestep_stats = defaultdict(lambda: {"content": 0, "style": 0, "lighting": 0})
    layer_stats = defaultdict(lambda: {"content": 0, "style": 0, "lighting": 0})
    selection_matrix = {}

    for step in range(num_steps):
        timestep = (step + 1) * 1000
        t = timestep / sum_timesteps

        selection_matrix[step] = {}

        # Time-based scaling (same for all layers at this timestep)
        scale_content = alpha * (1 - t) + beta
        scale_style = alpha * t + beta
        scale_lighting = gamma * t + beta

        if pattern == "s*":
            scale_content = scale_content % alpha if alpha > 0 else scale_content
            scale_style = scale_style % alpha if alpha > 0 else scale_style
            scale_lighting = scale_lighting % gamma if gamma > 0 else scale_lighting

        for layer_idx in range(num_layers):
            top_k_sum1, top_k_sum2, top_k_sum3 = layer_topk_sums[layer_idx]

            # Compute scores
            score_content = top_k_sum1 * scale_content
            score_style = top_k_sum2 * scale_style
            score_lighting = top_k_sum3 * scale_lighting

            # Determine winner
            scores = [score_content, score_style, score_lighting]
            winner_idx = scores.index(max(scores))
            winner = ["content", "style", "lighting"][winner_idx]

            # Record stats
            timestep_stats[step][winner] += 1
            layer_stats[layer_idx][winner] += 1
            selection_matrix[step][layer_idx] = winner

    return timestep_stats, layer_stats, selection_matrix, num_layers


def print_results(timestep_stats, layer_stats, selection_matrix, num_layers, num_steps, args):
    """Print analysis results."""

    print("\n" + "="*60)
    print("RESULTS: LoRA Selection per Timestep")
    print("="*60)
    print(f"{'Step':<6} {'Content':<12} {'Style':<12} {'Lighting':<12} {'Winner':<10}")
    print("-"*60)

    for step in range(num_steps):
        stats = timestep_stats[step]
        total = sum(stats.values())
        content_pct = stats['content'] / total * 100
        style_pct = stats['style'] / total * 100
        lighting_pct = stats['lighting'] / total * 100

        # Determine majority winner for this timestep
        winner = max(stats, key=stats.get)

        print(f"{step:<6} {stats['content']:>4} ({content_pct:>5.1f}%) "
              f"{stats['style']:>4} ({style_pct:>5.1f}%) "
              f"{stats['lighting']:>4} ({lighting_pct:>5.1f}%) "
              f"{winner:<10}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    total_content = sum(s['content'] for s in timestep_stats.values())
    total_style = sum(s['style'] for s in timestep_stats.values())
    total_lighting = sum(s['lighting'] for s in timestep_stats.values())
    total = total_content + total_style + total_lighting

    print(f"Total Content selections:  {total_content:>6} ({total_content/total*100:.1f}%)")
    print(f"Total Style selections:    {total_style:>6} ({total_style/total*100:.1f}%)")
    print(f"Total Lighting selections: {total_lighting:>6} ({total_lighting/total*100:.1f}%)")

    # Early vs Late analysis
    mid_step = num_steps // 2
    early_content = sum(timestep_stats[s]['content'] for s in range(mid_step))
    early_style = sum(timestep_stats[s]['style'] for s in range(mid_step))
    early_lighting = sum(timestep_stats[s]['lighting'] for s in range(mid_step))

    late_content = sum(timestep_stats[s]['content'] for s in range(mid_step, num_steps))
    late_style = sum(timestep_stats[s]['style'] for s in range(mid_step, num_steps))
    late_lighting = sum(timestep_stats[s]['lighting'] for s in range(mid_step, num_steps))

    print(f"\nEarly steps (0-{mid_step-1}):")
    early_total = early_content + early_style + early_lighting
    print(f"  Content: {early_content/early_total*100:.1f}%, Style: {early_style/early_total*100:.1f}%, Lighting: {early_lighting/early_total*100:.1f}%")

    print(f"\nLate steps ({mid_step}-{num_steps-1}):")
    late_total = late_content + late_style + late_lighting
    print(f"  Content: {late_content/late_total*100:.1f}%, Style: {late_style/late_total*100:.1f}%, Lighting: {late_lighting/late_total*100:.1f}%")


def main():
    args = parse_args()

    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print("Loading LoRA weights...")

    # Load LoRAs
    content_lora = record_content_loras[args.content_index]
    style_lora = record_style_loras[args.style_index]
    lighting_lora = record_lighting_loras[args.lighting_index]

    print(f"  Content: {content_lora}")
    print(f"  Style: {style_lora}")
    print(f"  Lighting: {lighting_lora}")

    content_weights = get_lora_weights(
        content_lora,
        sub_lora_weights_name=content_lora_weight_names[args.content_index]
    )
    style_weights = get_lora_weights(
        style_lora,
        sub_lora_weights_name=style_lora_weight_names[args.style_index]
    )
    lighting_weights = get_lora_weights(
        lighting_lora,
        sub_lora_weights_name=lighting_lora_weight_names[args.lighting_index]
    )

    # Extract layer pairs and move to GPU
    content_layers = get_layer_pairs(content_weights, device)
    style_layers = get_layer_pairs(style_weights, device)
    lighting_layers = get_layer_pairs(lighting_weights, device)

    # Run simulation
    timestep_stats, layer_stats, selection_matrix, num_layers = simulate_selection(
        content_layers, style_layers, lighting_layers, args
    )

    # Print results
    print_results(timestep_stats, layer_stats, selection_matrix, num_layers, args.num_steps, args)

    # Save to JSON if requested
    if args.output_json:
        results = {
            "config": {
                "content_index": args.content_index,
                "style_index": args.style_index,
                "lighting_index": args.lighting_index,
                "alpha": args.alpha,
                "beta": args.beta,
                "gamma": args.gamma,
                "num_steps": args.num_steps,
            },
            "timestep_stats": dict(timestep_stats),
            "layer_stats": {str(k): v for k, v in layer_stats.items()},
        }
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
