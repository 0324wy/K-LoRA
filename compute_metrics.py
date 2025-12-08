"""
Compute evaluation metrics for K-LoRA generated images:
- CLIP Score: Text-image alignment
- DINO Score: Image similarity (structural)
- Style Similarity: Style transfer quality
"""

import argparse
import os
import json
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, AutoProcessor, AutoModel
from torchvision import transforms
import numpy as np
from glob import glob

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_clip_model():
    """Load CLIP model for text-image similarity."""
    print("Loading CLIP model...")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    return model, processor


def load_dino_model():
    """Load DINOv2 model for image similarity."""
    print("Loading DINOv2 model...")
    model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
    processor = AutoProcessor.from_pretrained("facebook/dinov2-base")
    return model, processor


def compute_clip_score(model, processor, image_path, text):
    """Compute CLIP score between image and text."""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        # Normalize and compute cosine similarity
        image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
        score = (image_embeds @ text_embeds.T).item()

    return score


def compute_dino_similarity(model, processor, image_path1, image_path2):
    """Compute DINO similarity between two images."""
    image1 = Image.open(image_path1).convert("RGB")
    image2 = Image.open(image_path2).convert("RGB")

    inputs1 = processor(images=image1, return_tensors="pt").to(device)
    inputs2 = processor(images=image2, return_tensors="pt").to(device)

    with torch.no_grad():
        features1 = model(**inputs1).last_hidden_state.mean(dim=1)
        features2 = model(**inputs2).last_hidden_state.mean(dim=1)

        # Normalize and compute cosine similarity
        features1 = features1 / features1.norm(dim=-1, keepdim=True)
        features2 = features2 / features2.norm(dim=-1, keepdim=True)
        score = (features1 @ features2.T).item()

    return score


def compute_style_similarity(model, processor, generated_path, style_only_path):
    """Compute style similarity using CLIP image features."""
    image1 = Image.open(generated_path).convert("RGB")
    image2 = Image.open(style_only_path).convert("RGB")

    inputs = processor(images=[image1, image2], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        score = (image_features[0] @ image_features[1]).item()

    return score


def parse_args():
    parser = argparse.ArgumentParser(description="Compute evaluation metrics")
    parser.add_argument("--output_folder", type=str, required=True,
                        help="Folder containing comparison mode outputs")
    parser.add_argument("--content_prompt", type=str, default="eric cat",
                        help="Content trigger word")
    parser.add_argument("--style_prompt", type=str, default="SRFNGV01",
                        help="Style trigger word")
    parser.add_argument("--lighting_prompt", type=str, default="cinematic_octane",
                        help="Lighting trigger word")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Save results to JSON file")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load models
    clip_model, clip_processor = load_clip_model()
    dino_model, dino_processor = load_dino_model()

    # Find all seeds
    content_only_files = sorted(glob(os.path.join(args.output_folder, "content_only_seed*.png")))
    seeds = [f.split("seed")[-1].replace(".png", "") for f in content_only_files]

    if not seeds:
        print(f"No images found in {args.output_folder}")
        return

    print(f"\nFound {len(seeds)} seeds: {seeds}")

    # Prepare prompts
    full_prompt = f"{args.content_prompt} in {args.style_prompt} style, {args.lighting_prompt}"
    content_style_prompt = f"{args.content_prompt} in {args.style_prompt} style"

    # Results storage
    results = {
        "clip_scores": {"content_only": [], "style_only": [], "lighting_only": [],
                       "content_style": [], "content_style_lighting": []},
        "dino_content": {"content_style": [], "content_style_lighting": []},
        "dino_style": {"content_style": [], "content_style_lighting": []},
        "style_sim": {"content_style": [], "content_style_lighting": []},
    }

    print("\nComputing metrics...")
    print("="*60)

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")

        # File paths
        content_only = os.path.join(args.output_folder, f"content_only_seed{seed}.png")
        style_only = os.path.join(args.output_folder, f"style_only_seed{seed}.png")
        lighting_only = os.path.join(args.output_folder, f"lighting_only_seed{seed}.png")
        content_style = os.path.join(args.output_folder, f"content_style_seed{seed}.png")
        content_style_lighting = os.path.join(args.output_folder, f"content_style_lighting_seed{seed}.png")

        # CLIP Scores (text-image alignment)
        clip_content = compute_clip_score(clip_model, clip_processor, content_only, args.content_prompt)
        clip_style = compute_clip_score(clip_model, clip_processor, style_only, args.style_prompt)
        clip_lighting = compute_clip_score(clip_model, clip_processor, lighting_only, args.lighting_prompt)
        clip_cs = compute_clip_score(clip_model, clip_processor, content_style, content_style_prompt)
        clip_csl = compute_clip_score(clip_model, clip_processor, content_style_lighting, full_prompt)

        results["clip_scores"]["content_only"].append(clip_content)
        results["clip_scores"]["style_only"].append(clip_style)
        results["clip_scores"]["lighting_only"].append(clip_lighting)
        results["clip_scores"]["content_style"].append(clip_cs)
        results["clip_scores"]["content_style_lighting"].append(clip_csl)

        print(f"  CLIP Scores:")
        print(f"    Content Only:  {clip_content:.4f}")
        print(f"    Style Only:    {clip_style:.4f}")
        print(f"    Lighting Only: {clip_lighting:.4f}")
        print(f"    Content+Style: {clip_cs:.4f}")
        print(f"    Full 3-LoRA:   {clip_csl:.4f}")

        # DINO Scores (structural similarity to content)
        dino_cs_content = compute_dino_similarity(dino_model, dino_processor, content_style, content_only)
        dino_csl_content = compute_dino_similarity(dino_model, dino_processor, content_style_lighting, content_only)

        results["dino_content"]["content_style"].append(dino_cs_content)
        results["dino_content"]["content_style_lighting"].append(dino_csl_content)

        print(f"  DINO (vs Content):")
        print(f"    Content+Style: {dino_cs_content:.4f}")
        print(f"    Full 3-LoRA:   {dino_csl_content:.4f}")

        # Style Similarity (CLIP image features)
        style_sim_cs = compute_style_similarity(clip_model, clip_processor, content_style, style_only)
        style_sim_csl = compute_style_similarity(clip_model, clip_processor, content_style_lighting, style_only)

        results["style_sim"]["content_style"].append(style_sim_cs)
        results["style_sim"]["content_style_lighting"].append(style_sim_csl)

        print(f"  Style Similarity:")
        print(f"    Content+Style: {style_sim_cs:.4f}")
        print(f"    Full 3-LoRA:   {style_sim_csl:.4f}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY (Mean ± Std)")
    print("="*60)

    print("\n📊 CLIP Scores (Text-Image Alignment):")
    for mode, scores in results["clip_scores"].items():
        if scores:
            mean = np.mean(scores)
            std = np.std(scores)
            print(f"  {mode:25s}: {mean:.4f} ± {std:.4f}")

    print("\n📊 DINO Scores (Content Preservation):")
    for mode, scores in results["dino_content"].items():
        if scores:
            mean = np.mean(scores)
            std = np.std(scores)
            print(f"  {mode:25s}: {mean:.4f} ± {std:.4f}")

    print("\n📊 Style Similarity:")
    for mode, scores in results["style_sim"].items():
        if scores:
            mean = np.mean(scores)
            std = np.std(scores)
            print(f"  {mode:25s}: {mean:.4f} ± {std:.4f}")

    # Save to JSON if requested
    if args.output_json:
        # Compute summary statistics
        summary = {
            "config": {
                "output_folder": args.output_folder,
                "content_prompt": args.content_prompt,
                "style_prompt": args.style_prompt,
                "lighting_prompt": args.lighting_prompt,
                "num_seeds": len(seeds),
            },
            "raw_scores": results,
            "summary": {
                "clip_scores": {},
                "dino_content": {},
                "style_sim": {},
            }
        }

        for mode, scores in results["clip_scores"].items():
            if scores:
                summary["summary"]["clip_scores"][mode] = {
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores)),
                }

        for mode, scores in results["dino_content"].items():
            if scores:
                summary["summary"]["dino_content"][mode] = {
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores)),
                }

        for mode, scores in results["style_sim"].items():
            if scores:
                summary["summary"]["style_sim"][mode] = {
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores)),
                }

        with open(args.output_json, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
