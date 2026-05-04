#!/usr/bin/env python3
# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

import torch

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

try:
    from safetensors.torch import load_file as safe_load_file
    from safetensors.torch import save_file as safe_save_file
except Exception:  # pragma: no cover
    safe_load_file = None
    safe_save_file = None

from diffusers.models.transformers import NiTTransformer2DModel
from diffusers.schedulers import NiTFlowMatchScheduler


MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    "nit-s": {
        "depth": 12,
        "hidden_size": 384,
        "num_heads": 6,
        "encoder_depth": 4,
        "projector_dim": 768,
        "z_dim": 768,
    },
    "nit-b": {
        "depth": 12,
        "hidden_size": 768,
        "num_heads": 12,
        "encoder_depth": 4,
        "z_dim": 1280,
    },
    "nit-l": {
        "depth": 24,
        "hidden_size": 1024,
        "num_heads": 16,
        "encoder_depth": 6,
        "z_dim": 1280,
    },
    "nit-xl": {
        "depth": 28,
        "hidden_size": 1152,
        "num_heads": 16,
        "encoder_depth": 8,
        "z_dim": 1280,
    },
    "nit-xxl": {
        "depth": 40,
        "hidden_size": 1536,
        "num_heads": 24,
        "encoder_depth": 8,
        "z_dim": 1280,
        "use_adaln_lora": True,
        "adaln_lora_dim": 512,
    },
}


def _load_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    if checkpoint_path.endswith(".safetensors"):
        if safe_load_file is None:
            raise ImportError("Install safetensors to convert .safetensors checkpoints.")
        state_dict = safe_load_file(checkpoint_path, device="cpu")
    else:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state_dict, dict):
            for key in ("state_dict", "model", "module"):
                if key in state_dict and isinstance(state_dict[key], dict):
                    state_dict = state_dict[key]
                    break
    return _clean_state_dict(state_dict)


def _clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}
    prefixes = ("model.", "module.", "transformer.")
    for key, value in state_dict.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
        cleaned[key] = value
    return cleaned


def _save_config(output_dir: Path, config: Dict[str, Any]):
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
        f.write("\n")


def _save_weights(output_dir: Path, state_dict: Dict[str, torch.Tensor], safe_serialization: bool):
    output_dir.mkdir(parents=True, exist_ok=True)
    if safe_serialization:
        if safe_save_file is None:
            raise ImportError("Install safetensors or pass --no-safe-serialization.")
        safe_save_file(state_dict, str(output_dir / "diffusion_pytorch_model.safetensors"), metadata={"format": "pt"})
    else:
        torch.save(state_dict, output_dir / "diffusion_pytorch_model.bin")


def _write_model_index(output_dir: Path, vae: str | None):
    model_index = {
        "_class_name": "NiTPipeline",
        "_diffusers_version": "0.30.1",
        "scheduler": ["diffusers", "NiTFlowMatchScheduler"],
        "transformer": ["diffusers", "NiTTransformer2DModel"],
    }
    if vae is not None:
        model_index["vae"] = ["diffusers", "AutoencoderDC" if "dc-ae" in vae else "AutoencoderKL"]
    with open(output_dir / "model_index.json", "w", encoding="utf-8") as f:
        json.dump(model_index, f, indent=2, sort_keys=True)
        f.write("\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert original NiT checkpoints to a Diffusers pipeline directory.")
    parser.add_argument("--checkpoint", required=True, help="Path to an original NiT .safetensors/.bin/.pt checkpoint.")
    parser.add_argument("--output", required=True, help="Output Diffusers model directory.")
    parser.add_argument("--model-size", choices=sorted(MODEL_PRESETS), default="nit-xl")
    parser.add_argument("--vae", default="mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers")
    parser.add_argument("--copy-vae", default=None, help="Optional local VAE directory to copy into output/vae.")
    parser.add_argument("--mode", choices=["ode", "sde"], default="ode")
    parser.add_argument("--path-type", choices=["linear", "cosine"], default="linear")
    parser.add_argument("--input-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=1)
    parser.add_argument("--in-channels", type=int, default=32)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--class-dropout-prob", type=float, default=0.1)
    parser.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--safe-serialization", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--check-load", action="store_true", help="Instantiate the converted transformer and load weights.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output)
    transformer_dir = output_dir / "transformer"
    scheduler_dir = output_dir / "scheduler"

    state_dict = _load_state_dict(args.checkpoint)
    config = {
        "input_size": args.input_size,
        "patch_size": args.patch_size,
        "in_channels": args.in_channels,
        "class_dropout_prob": args.class_dropout_prob,
        "num_classes": args.num_classes,
        "qk_norm": args.qk_norm,
        **MODEL_PRESETS[args.model_size],
    }

    if args.check_load:
        model = NiTTransformer2DModel(**config)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys or unexpected_keys:
            print("Missing keys:", missing_keys)
            print("Unexpected keys:", unexpected_keys)
            raise SystemExit(1)

    _save_config(transformer_dir, {"_class_name": "NiTTransformer2DModel", **config})
    _save_weights(transformer_dir, state_dict, args.safe_serialization)

    _save_config(
        scheduler_dir,
        {
            "_class_name": "NiTFlowMatchScheduler",
            "mode": args.mode,
            "path_type": args.path_type,
            "num_train_timesteps": 1000,
        },
    )

    if args.copy_vae is not None:
        target_vae_dir = output_dir / "vae"
        if target_vae_dir.exists():
            shutil.rmtree(target_vae_dir)
        shutil.copytree(args.copy_vae, target_vae_dir)
    elif args.vae:
        with open(output_dir / "vae_pretrained_model_name_or_path.txt", "w", encoding="utf-8") as f:
            f.write(args.vae + os.linesep)

    _write_model_index(output_dir, args.vae)
    print(f"Saved Diffusers-style NiT pipeline to {output_dir}")


if __name__ == "__main__":
    main()
