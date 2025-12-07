import argparse
from diffusers import DiffusionPipeline, FluxTransformer2DModel
import torch
import os
from utils import insert_community_flux_lora_to_unet, insert_community_flux_lora_to_unet_3lora
from klora import KLoRALinearLayer, glo_count
import klora


def set_lora_forward_type(pipe, forward_type):
    """Set forward_type for all KLoRALinearLayer instances in the pipeline.

    Args:
        pipe: The diffusion pipeline
        forward_type: One of "merge", "weight_1", "weight_2", "weight_3"
    """
    for module in pipe.transformer.modules():
        if isinstance(module, KLoRALinearLayer):
            module.forward_type = forward_type


def reset_glo_count():
    """Reset the global counter for K-LoRA timestep tracking."""
    klora.glo_count = 0


record_content_loras = [
    "ginipick/flux-lora-eric-cat",
    "glif-loradex-trainer/antix82_flux_dev_marv_simplecap_v1",
    "glif-loradex-trainer/festerbitcoin_86601_cats",
    "glif-loradex-trainer/fabian3000_chillguy",
]
content_triggers = [
    "eric cat",
    "marv frog man marv",
    "Cat rule the world, ",
    "chillguy",
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
    "glif-loradex-trainer/maxxd4240_SketchOnWater",
    "glif-loradex-trainer/araminta_k_flux_dev_leonardlesliebrookes",
    "glif-loradex-trainer/araminta_k_flux_dev_karl_weiner",
    "glif-loradex-trainer/araminta_k_flux_dev_tarot_test_1",
    "glif-loradex-trainer/i12_appelsiensam_fanimals_v1",
    "glif-loradex-trainer/goldenark__WaterColorSketchStyle",
    "glif-loradex-trainer/fabian3000_henrymajor",
    "glif-loradex-trainer/fabian3000_impressionism2",
]
style_triggers = [
    "SRFNGV01",
    "Acryl!ck",
    "swrind",
    "BluD!!",
    "SkeWat, water color sketch style",
    "illustration style",
    "collage style",
    "illustration style",
    "FNMLS_PPLSNSM",
    "WaterColorSketchStyle",
    "henrymajorstyle",
    "impressionist",
]
style_lora_weight_names = [
    "flux_surf.safetensors",
    "AcrylicWorld.safetensors",
    "Seiwert_Industrial_v1.safetensors",
    "BlueDraw.safetensors",
    "SketchOnWater.safetensors",
    "flux_dev_leonardlesliebrookes.safetensors",
    "flux_dev_karl_weiner.safetensors",
    "flux_dev_tarot_test_1.safetensors",
    "appelsiensam_fanimals_v1.safetensors",
    "WaterColorSketchStyle.safetensors",
    "henrymajor.safetensors",
    "impressionism2.safetensors",
]

# Lighting/Atmosphere LoRAs for 3-LoRA fusion
record_lighting_loras = [
    "aixonlab/FLUX.1-dev-LoRA-Cinematic-Octane",
    "fofr/flux-80s-cyberpunk",
]
lighting_triggers = [
    "cinematic_octane",
    "style of 80s cyberpunk",
]
lighting_lora_weight_names = [
    "cinematic-octane.safetensors",
    "lora.safetensors",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="black-forest-labs/FLUX.1-dev",
        help="Pretrained model path or HuggingFace model ID",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        help="Output folder path",
        default="output/",
    )
    parser.add_argument(
        "--content_index",
        type=str,
        default="0",
    )
    parser.add_argument(
        "--style_index",
        type=str,
        help="Output folder path",
        default="0",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        help="Pattern for the image generation",
        default="s*",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        help="Number of images to generate",
        default=40,
    )
    parser.add_argument(
        "--lighting_index",
        type=str,
        help="Index for lighting/atmosphere LoRA (enables 3-LoRA mode)",
        default=None,
    )
    parser.add_argument(
        "--gamma",
        type=float,
        help="Gamma parameter for lighting LoRA time scaling",
        default=0.3,
    )
    parser.add_argument(
        "--comparison_mode",
        action="store_true",
        help="Generate 5 comparison images per seed (ablation study)",
    )
    return parser.parse_args()


args = parse_args()
pattern = args.pattern
if pattern == "s*":
    alpha = 1.5
    beta = alpha * 0.85
else:
    alpha = 1.5
    beta = 0.5
gamma = args.gamma
flux_diffuse_step = 28

# Check for 3-LoRA mode
use_3_loras = args.lighting_index is not None

content_lora = record_content_loras[int(args.content_index)]
style_lora = record_style_loras[int(args.style_index)]
content_trigger_word = content_triggers[int(args.content_index)]
style_trigger_word = style_triggers[int(args.style_index)]
content_lora_weight_name = content_lora_weight_names[int(args.content_index)]
style_lora_weight_name = style_lora_weight_names[int(args.style_index)]

pipe = DiffusionPipeline.from_pretrained(
    args.pretrained_model_name_or_path, torch_dtype=torch.bfloat16
)

if use_3_loras:
    # 3-LoRA mode: Content + Style + Lighting
    lighting_lora = record_lighting_loras[int(args.lighting_index)]
    lighting_trigger_word = lighting_triggers[int(args.lighting_index)]
    lighting_lora_weight_name = lighting_lora_weight_names[int(args.lighting_index)]

    print(f"3-LoRA Mode: Content={content_lora}, Style={style_lora}, Lighting={lighting_lora}")

    unet = insert_community_flux_lora_to_unet_3lora(
        unet=pipe,
        lora_weights_content_path=content_lora,
        lora_weights_style_path=style_lora,
        lora_weights_lighting_path=lighting_lora,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        diffuse_step=flux_diffuse_step,
        content_lora_weight_name=content_lora_weight_name,
        style_lora_weight_name=style_lora_weight_name,
        lighting_lora_weight_name=lighting_lora_weight_name,
    )

    prompt = f"{content_trigger_word} in {style_trigger_word} style, {lighting_trigger_word}."
else:
    # 2-LoRA mode: Content + Style (backward compatible)
    print(f"2-LoRA Mode: Content={content_lora}, Style={style_lora}")

    unet = insert_community_flux_lora_to_unet(
        unet=pipe,
        lora_weights_content_path=content_lora,
        lora_weights_style_path=style_lora,
        alpha=alpha,
        beta=beta,
        diffuse_step=flux_diffuse_step,
        content_lora_weight_name=content_lora_weight_name,
        style_lora_weight_name=style_lora_weight_name,
    )

    prompt = content_trigger_word + " in " + style_trigger_word + " style."

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe.to(device, dtype=torch.float16)


def run():
    seeds = list(range(args.num_samples))
    os.makedirs(args.output_folder, exist_ok=True)

    if args.comparison_mode and use_3_loras:
        # Comparison mode: generate 5 images per seed
        comparison_modes = [
            ("content_only", "weight_1", content_trigger_word),
            ("style_only", "weight_2", style_trigger_word),
            ("lighting_only", "weight_3", lighting_trigger_word),
            ("content_style", "merge_2lora", f"{content_trigger_word} in {style_trigger_word} style."),
            ("content_style_lighting", "merge", prompt),
        ]

        for seed in seeds:
            print(f"\n=== Seed {seed} ===")
            for mode_name, forward_type, mode_prompt in comparison_modes:
                # Reset global counter for each image
                reset_glo_count()

                # Set the forward type for all K-LoRA layers
                set_lora_forward_type(pipe, forward_type)

                generator = torch.Generator(device=device).manual_seed(seed)
                image = pipe(prompt=mode_prompt, generator=generator).images[0]

                output_path = os.path.join(args.output_folder, f"{mode_name}_seed{seed}.png")
                print(f"  [{mode_name}] Saving to {output_path}")
                image.save(output_path)

        print(f"\nComparison mode complete! Generated {len(seeds) * 5} images.")

    elif args.comparison_mode and not use_3_loras:
        print("Warning: --comparison_mode requires --lighting_index to be set.")
        print("Running in normal 2-LoRA mode instead.")

        for index, seed in enumerate(seeds):
            generator = torch.Generator(device=device).manual_seed(seed)
            image = pipe(prompt=prompt, generator=generator).images[0]
            output_path = os.path.join(args.output_folder, f"output_image_{index}.png")
            print(f"Saving output to {output_path}")
            image.save(output_path)

    else:
        # Normal mode: single image per seed
        for index, seed in enumerate(seeds):
            reset_glo_count()
            generator = torch.Generator(device=device).manual_seed(seed)
            image = pipe(prompt=prompt, generator=generator).images[0]
            output_path = os.path.join(args.output_folder, f"output_image_{index}.png")
            print(f"Saving output to {output_path}")
            image.save(output_path)


if __name__ == "__main__":
    run()
