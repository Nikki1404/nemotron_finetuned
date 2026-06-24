#!/usr/bin/env python3

import argparse
import types
from pathlib import Path

import torch
import lightning.pytorch as pl
from omegaconf import OmegaConf
import nemo.collections.asr as nemo_asr


def freeze_decoder(model):
    for p in model.parameters():
        p.requires_grad = False

    for module_name in ["decoder", "joint", "prompt_kernel"]:
        if hasattr(model, module_name):
            for p in getattr(model, module_name).parameters():
                p.requires_grad = True

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(
        f"[params] total={total:,} "
        f"trainable={trainable:,} "
        f"({100 * trainable / total:.2f}%)"
    )


def patch_training_step(model, prompt_index):
    old_training_step = model.training_step

    def new_training_step(self, batch, batch_idx):
        if len(batch) == 4:
            signal, signal_len, transcript, transcript_len = batch

            prompt_indices = torch.full(
                (signal.shape[0],),
                prompt_index,
                dtype=torch.long,
                device=signal.device,
            )

            batch = (
                signal,
                signal_len,
                transcript,
                transcript_len,
                prompt_indices,
            )

        return old_training_step(batch, batch_idx)

    model.training_step = types.MethodType(
        new_training_step,
        model,
    )

    print(
        f"[patch] Added prompt index {prompt_index}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-manifest",
        required=True,
    )

    parser.add_argument(
        "--base-model",
        required=True,
    )

    parser.add_argument(
        "--output-nemo",
        required=True,
    )

    parser.add_argument(
        "--language",
        default="en-US",
    )

    parser.add_argument(
        "--max-epochs",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=5e-6,
    )

    args = parser.parse_args()

    print("[load]", args.base_model)

    model = nemo_asr.models.ASRModel.restore_from(
        args.base_model
    )

    try:
        model.set_default_prompt(args.language)
    except:
        pass

    try:
        model.set_inference_prompt(args.language)
    except:
        pass

    prompt_index = int(
        model.cfg.train_ds.prompt_dictionary[
            args.language
        ]
    )

    print(
        f"[language] {args.language} "
        f"prompt={prompt_index}"
    )

    freeze_decoder(model)

    patch_training_step(
        model,
        prompt_index,
    )

    train_cfg = OmegaConf.create(
        {
            "manifest_filepath":
                args.train_manifest,
            "sample_rate": 16000,
            "batch_size":
                args.batch_size,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": True,
            "use_lhotse": False,
            "max_duration": 9999,
            "min_duration": 0.1,
        }
    )

    model.setup_training_data(
        train_cfg
    )

    model.cfg.optim = OmegaConf.create(
        {
            "name": "adamw",
            "lr": args.lr,
            "betas": [0.9, 0.98],
            "weight_decay": 0.001,
        }
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=args.max_epochs,
        precision="bf16-mixed",
        logger=False,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        limit_val_batches=0,
        check_val_every_n_epoch=None,
    )

    print(
        "[train] Starting fine-tuning..."
    )

    trainer.fit(model)

    Path(
        args.output_nemo
    ).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_to(
        args.output_nemo
    )

    print(
        f"[saved] {args.output_nemo}"
    )


if __name__ == "__main__":
    main()
