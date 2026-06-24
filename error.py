#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import types
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

    try:
        model.set_default_prompt(language)
    except Exception as e:
        print(f"[warn] set_default_prompt({language}) failed: {e}")

    return model


def set_freeze_mode(model: Any, freeze_mode: str) -> None:
    if freeze_mode == "none":
        for p in model.parameters():
            p.requires_grad = True
        return

    if freeze_mode == "decoder_only":
        for p in model.parameters():
            p.requires_grad = False

        train_modules = ("decoder", "joint", "prompt_kernel")

        for name, module in model.named_modules():
            if any(k in name.lower() for k in train_modules):
                for p in module.parameters(recurse=True):
                    p.requires_grad = True
        return

    if freeze_mode == "last_encoder":
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
    print(f"[params] total={total:,} trainable={trainable:,} ({trainable / max(1, total) * 100:.2f}%)")


def get_prompt_index(model: Any, language: str) -> int:
    prompt_dict = model.cfg.train_ds.prompt_dictionary
    if language not in prompt_dict:
        raise ValueError(f"Language {language} not in prompt dictionary")
    return int(prompt_dict[language])


def patch_batch_prompt_indices(model: Any, prompt_index: int) -> None:
    old_training_step = model.training_step
    old_validation_step = model.validation_step

    def add_prompt(batch):
        if isinstance(batch, (tuple, list)) and len(batch) == 4:
            signal, signal_len, transcript, transcript_len = batch

            import torch

            prompt_indices = torch.full(
                (signal.shape[0],),
                prompt_index,
                dtype=torch.long,
                device=signal.device,
            )

            return signal, signal_len, transcript, transcript_len, prompt_indices

        return batch

    def new_training_step(self, batch, batch_idx):
        batch = add_prompt(batch)
        return old_training_step(batch, batch_idx)

    def new_validation_step(self, batch, batch_idx, dataloader_idx=0):
        batch = add_prompt(batch)
        try:
            return old_validation_step(batch, batch_idx, dataloader_idx)
        except TypeError:
            return old_validation_step(batch, batch_idx)

    model.training_step = types.MethodType(new_training_step, model)
    model.validation_step = types.MethodType(new_validation_step, model)

    print(f"[patch] Added prompt_indices={prompt_index} when batch has only 4 items")


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--train-manifest", required=True)
    ap.add_argument("--val-manifest", required=True)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--output-nemo", required=True)

    ap.add_argument("--language", default="en-US")
    ap.add_argument("--freeze-mode", default="decoder_only", choices=["decoder_only", "last_encoder", "none"])
    ap.add_argument("--max-epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--devices", type=int, default=1)
    ap.add_argument("--precision", default="bf16-mixed")
    ap.add_argument("--num-workers", type=int, default=0)

    args = ap.parse_args()

    import torch
    import lightning.pytorch as pl
    from omegaconf import OmegaConf

    model = load_model(args.base_model, args.language)

    prompt_index = get_prompt_index(model, args.language)
    print(f"[language] {args.language} prompt_index={prompt_index}")

    set_freeze_mode(model, args.freeze_mode)
    count_trainable(model)

    patch_batch_prompt_indices(model, prompt_index)

    train_cfg = OmegaConf.create({
        "manifest_filepath": str(Path(args.train_manifest).resolve()),
        "sample_rate": 16000,
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "max_duration": 9999.0,
        "min_duration": 0.1,
        "is_tarred": False,
        "use_lhotse": False,
    })

    val_cfg = OmegaConf.create({
        "manifest_filepath": str(Path(args.val_manifest).resolve()),
        "sample_rate": 16000,
        "batch_size": 1,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "max_duration": 9999.0,
        "min_duration": 0.1,
        "is_tarred": False,
        "use_lhotse": False,
    })

    model.setup_training_data(train_data_config=train_cfg)
    model.setup_validation_data(val_data_config=val_cfg)

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
        num_sanity_val_steps=0,
    )

    model.set_trainer(trainer)

    print("[train] Starting fine-tuning...")
    trainer.fit(model)

    output = Path(args.output_nemo)
    output.parent.mkdir(parents=True, exist_ok=True)

    model.save_to(str(output))
    print(f"[done] Fine-tuned model saved to: {output}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
