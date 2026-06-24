#!/usr/bin/env python3
"""
Fine-tune Nemotron 3.5 ASR Streaming 0.6B on a small NeMo manifest.

This script is intentionally conservative for small data:
  - decoder_only mode freezes encoder parameters
  - low LR
  - batch size 1
  - saves a new .nemo model
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any


def load_model(model_path: str, language: str):
    import nemo.collections.asr as nemo_asr

    cls = nemo_asr.models.EncDecRNNTBPEModelWithPrompt
    if model_path.endswith(".nemo") or Path(model_path).exists():
        model = cls.restore_from(model_path, map_location="cpu")
    else:
        model = cls.from_pretrained(model_path, map_location="cpu")

    try:
        model.set_inference_prompt(language)
    except Exception as e:
        print(f"[warn] set_inference_prompt({language}) failed: {e}")
    return model


def set_freeze_mode(model: Any, freeze_mode: str) -> None:
    if freeze_mode == "none":
        for p in model.parameters():
            p.requires_grad = True
        return

    if freeze_mode == "decoder_only":
        # Freeze everything first
        for p in model.parameters():
            p.requires_grad = False
        # Then unfreeze decoder/joint/prediction-style modules if present
        names_to_train = ("decoder", "joint", "loss", "wer")
        for name, module in model.named_modules():
            if any(k in name.lower() for k in names_to_train):
                for p in module.parameters(recurse=False):
                    p.requires_grad = True
        # Safer fallback: if no params got enabled, unfreeze non-encoder params
        if not any(p.requires_grad for p in model.parameters()):
            for name, p in model.named_parameters():
                if not name.startswith("encoder."):
                    p.requires_grad = True
        return

    if freeze_mode == "last_encoder":
        # Train decoder/joint + last 20% encoder params by layer order where possible.
        set_freeze_mode(model, "decoder_only")
        enc_params = [(n, p) for n, p in model.named_parameters() if n.startswith("encoder.")]
        start = int(len(enc_params) * 0.8)
        for _, p in enc_params[start:]:
            p.requires_grad = True
        return

    raise ValueError(f"Unknown freeze_mode: {freeze_mode}")


def count_trainable(model: Any) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[params] total={total:,} trainable={trainable:,} ({trainable/max(1,total)*100:.2f}%)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-manifest", required=True)
    ap.add_argument("--val-manifest", required=True)
    ap.add_argument("--base-model", required=True, help="Base .nemo path or HF name")
    ap.add_argument("--output-nemo", required=True)
    ap.add_argument("--language", default="en-US")
    ap.add_argument("--freeze-mode", default="decoder_only", choices=["decoder_only", "last_encoder", "none"])
    ap.add_argument("--max-epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--devices", type=int, default=1)
    ap.add_argument("--precision", default="bf16-mixed")
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    import torch
    import lightning.pytorch as pl
    from omegaconf import OmegaConf

    model = load_model(args.base_model, args.language)
    set_freeze_mode(model, args.freeze_mode)
    count_trainable(model)

    # Conservative training/validation config. Avoids heavy bucketing complexity for tiny data.
    train_cfg = OmegaConf.create({
        "manifest_filepath": str(Path(args.train_manifest).resolve()),
        "sample_rate": 16000,
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "max_duration": 120.0,
        "min_duration": 0.1,
        "is_tarred": False,
    })
    val_cfg = OmegaConf.create({
        "manifest_filepath": str(Path(args.val_manifest).resolve()),
        "sample_rate": 16000,
        "batch_size": 1,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "max_duration": 120.0,
        "min_duration": 0.1,
        "is_tarred": False,
    })

    model.setup_training_data(train_data_config=train_cfg)
    model.setup_validation_data(val_data_config=val_cfg)

    # Set optimizer config expected by NeMo ASR models.
    model.cfg.optim = OmegaConf.create({
        "name": "adamw",
        "lr": args.lr,
        "betas": [0.9, 0.98],
        "weight_decay": 0.001,
        "sched": {
            "name": "CosineAnnealing",
            "warmup_steps": 5,
            "min_lr": args.lr / 20.0,
        },
    })

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices if torch.cuda.is_available() else 1,
        max_epochs=args.max_epochs,
        precision=args.precision if torch.cuda.is_available() else "32-true",
        gradient_clip_val=1.0,
        log_every_n_steps=1,
        enable_checkpointing=False,
    )

    model.set_trainer(trainer)
    print("[train] Starting fine-tuning...")
    trainer.fit(model)

    output = Path(args.output_nemo)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save_to(str(output))
    print(f"[done] Fine-tuned model saved to: {output}")


if __name__ == "__main__":
    # Prevent tokenizer parallelism warnings/noise in containers.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
