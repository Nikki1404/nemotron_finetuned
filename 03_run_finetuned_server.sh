#!/usr/bin/env bash
set -euo pipefail

# Run existing server image, but override MODEL_NAME to the fine-tuned model.
docker run --gpus all -it --rm \
  -p 8003:8003 \
  -v "$PWD/ft_models":/srv/models \
  -e MODEL_NAME=/srv/models/nemotron_inspira_decoder_ft.nemo \
  nemotron-asr
