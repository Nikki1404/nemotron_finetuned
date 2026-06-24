#!/usr/bin/env bash
set -euo pipefail

# Build your original image. This Dockerfile already downloads:
# /srv/nemotron-3.5-asr-streaming-0.6b.nemo

docker build -t nemotron-asr .

mkdir -p ft_models

# Enter same image for fine-tuning.
# /workspace = current project folder
# /srv/nemotron-3.5-asr-streaming-0.6b.nemo = base model inside image
# /srv/models = mapped to ./ft_models on host, so fine-tuned model is saved outside container

docker run --gpus all -it --rm \
  -v "$PWD":/workspace \
  -v "$PWD/ft_models":/srv/models \
  nemotron-asr \
  bash
