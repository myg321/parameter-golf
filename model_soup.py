#!/usr/bin/env python3
"""
model_soup.py: Universal Model Soup & Model Stock weight averaging for PEFT LoRA adapters.

Supports:
- Arbitrary number of input adapter checkpoints
- Uniform averaging (Uniform Soup) or Custom Weighted interpolation (Model Stock / Greedy Soup)
- Cosine similarity matrix computation across all input adapters and against the merged adapter
- Safe output generation compatible with merge_lora.py and eval_qwen.sh
"""

import argparse
import json
import os
import shutil
import sys
import torch
from safetensors.torch import load_file, save_file


def parse_args():
    parser = argparse.ArgumentParser(description="Universal Model Soup for LoRA Adapters")
    parser.add_argument(
        "--adapters",
        nargs="+",
        required=True,
        help="List of paths to adapter final directories (containing adapter_model.safetensors)",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="List of weights corresponding to each adapter (sums will be normalized to 1.0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory to save the merged adapter",
    )
    return parser.parse_args()


def compute_tensor_similarity(t1, t2):
    v1 = t1.flatten().float()
    v2 = t2.flatten().float()
    norm1 = torch.norm(v1)
    norm2 = torch.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 1.0
    return (torch.dot(v1, v2) / (norm1 * norm2)).item()


def main():
    args = parse_args()
    num_models = len(args.adapters)
    print(f"=== [Model Soup] Initializing fusion of {num_models} adapters ===")
    for i, p in enumerate(args.adapters):
        print(f"  Adapter [{i}]: {p}")

    if args.weights is not None:
        if len(args.weights) != num_models:
            raise ValueError(f"Number of weights ({len(args.weights)}) != number of adapters ({num_models})")
        total_w = sum(args.weights)
        weights = [w / total_w for w in args.weights]
    else:
        weights = [1.0 / num_models] * num_models

    print(f"[Model Soup] Using Normalized Weights: {[round(w, 4) for w in weights]}")

    state_dicts = []
    for i, p in enumerate(args.adapters):
        st_path = os.path.join(p, "adapter_model.safetensors")
        bin_path = os.path.join(p, "adapter_model.bin")
        if os.path.exists(st_path):
            print(f"[Model Soup] Loading adapter [{i}] from {st_path}...")
            sd = load_file(st_path, device="cpu")
        elif os.path.exists(bin_path):
            print(f"[Model Soup] Loading adapter [{i}] from {bin_path}...")
            sd = torch.load(bin_path, map_location="cpu")
        else:
            raise FileNotFoundError(f"No adapter_model.safetensors or .bin found in {p}")
        state_dicts.append(sd)

    # Check key consistency
    keys = list(state_dicts[0].keys())
    for i in range(1, num_models):
        curr_keys = list(state_dicts[i].keys())
        if set(keys) != set(curr_keys):
            diff1 = set(keys) - set(curr_keys)
            diff2 = set(curr_keys) - set(keys)
            raise ValueError(f"Key mismatch between adapter 0 and adapter {i}! Diff: {diff1 or diff2}")

    print(f"[Model Soup] All {len(keys)} keys matched perfectly across all {num_models} checkpoints.")

    # Compute pairwise similarities
    print("\n--- Pairwise Weight Vector Cosine Similarities ---")
    for i in range(num_models):
        for j in range(i + 1, num_models):
            sims = []
            for k in keys:
                s = compute_tensor_similarity(state_dicts[i][k], state_dicts[j][k])
                sims.append(s)
            avg_sim = sum(sims) / len(sims)
            import math
            clamped = min(1.0, max(-1.0, avg_sim))
            angle = math.acos(clamped) * 180.0 / math.pi
            print(f"  Similarity (Adapter {i} vs Adapter {j}): {avg_sim:.6f} (Angular distance: {angle:.2f}°)")

    # Fuse tensors
    print("\n--- Fusing adapter tensors via linear convex combination ---")
    fused_sd = {}
    for k in keys:
        fused_tensor = torch.zeros_like(state_dicts[0][k], dtype=torch.float32)
        for i in range(num_models):
            fused_tensor.add_(state_dicts[i][k].float(), alpha=weights[i])
        fused_sd[k] = fused_tensor.to(state_dicts[0][k].dtype)

    # Compute similarity between each component and merged soup
    print("\n--- Component vs Merged Soup Similarities ---")
    for i in range(num_models):
        sims = [compute_tensor_similarity(state_dicts[i][k], fused_sd[k]) for k in keys]
        print(f"  Similarity (Adapter {i} vs Merged Soup): {sum(sims)/len(sims):.6f}")

    # Save merged adapter
    os.makedirs(args.output, exist_ok=True)
    # Copy metadata files from adapter 0
    for fname in os.listdir(args.adapters[0]):
        if fname not in ["adapter_model.safetensors", "adapter_model.bin"]:
            src = os.path.join(args.adapters[0], fname)
            dst = os.path.join(args.output, fname)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
            elif os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)

    out_st = os.path.join(args.output, "adapter_model.safetensors")
    print(f"\n[Model Soup] Saving merged weights to {out_st}...")
    save_file(fused_sd, out_st)

    manifest = {
        "adapters": args.adapters,
        "weights": weights,
        "num_tensors": len(keys),
    }
    with open(os.path.join(args.output, "soup_manifest.json"), "w") as fp:
        json.dump(manifest, fp, indent=2)

    print(f"=== [Model Soup] Successfully created soup adapter at {args.output} ===\n")


if __name__ == "__main__":
    main()
