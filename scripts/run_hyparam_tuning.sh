#!/bin/bash
set -e

cd /workspace

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_HOME=/usr/local/cuda-12.4
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/nvvm/lib64:$LD_LIBRARY_PATH
unset NUMBA_CUDA_USE_NVIDIA_BINDING

BASE_MODEL="/srv/nemotron-3.5-asr-streaming-0.6b.nemo"
TRAIN_MANIFEST="data/manifests/train_aligned_aug_manifest.json"
VAL_MANIFEST="data/manifests/val_aligned_manifest.json"
TEST_MANIFEST="data/manifests/test_aligned_manifest.json"

mkdir -p /srv/models
mkdir -p results/hparam_tuning

echo "Evaluating base model..."
python3.11 scripts/evaluate_manifest.py \
  --model $BASE_MODEL \
  --manifest $TEST_MANIFEST \
  --language en-US \
  --output-jsonl results/hparam_tuning/base_eval.jsonl

echo "Training v1: lr=3e-6 epochs=2"
python3.11 scripts/finetune_nemotron.py \
  --train-manifest $TRAIN_MANIFEST \
  --val-manifest $VAL_MANIFEST \
  --base-model $BASE_MODEL \
  --output-nemo /srv/models/finetuned_nemotron_v1_lr3e6_ep2.nemo \
  --freeze-mode decoder_only \
  --max-epochs 2 \
  --batch-size 1 \
  --lr 3e-6 \
  --language en-US \
  --precision bf16-mixed

python3.11 scripts/evaluate_manifest.py \
  --model /srv/models/finetuned_nemotron_v1_lr3e6_ep2.nemo \
  --manifest $TEST_MANIFEST \
  --language en-US \
  --output-jsonl results/hparam_tuning/v1_eval.jsonl

echo "Training v2: lr=2e-6 epochs=3"
python3.11 scripts/finetune_nemotron.py \
  --train-manifest $TRAIN_MANIFEST \
  --val-manifest $VAL_MANIFEST \
  --base-model $BASE_MODEL \
  --output-nemo /srv/models/finetuned_nemotron_v2_lr2e6_ep3.nemo \
  --freeze-mode decoder_only \
  --max-epochs 3 \
  --batch-size 1 \
  --lr 2e-6 \
  --language en-US \
  --precision bf16-mixed

python3.11 scripts/evaluate_manifest.py \
  --model /srv/models/finetuned_nemotron_v2_lr2e6_ep3.nemo \
  --manifest $TEST_MANIFEST \
  --language en-US \
  --output-jsonl results/hparam_tuning/v2_eval.jsonl

echo "Training v3: lr=1e-6 epochs=3"
python3.11 scripts/finetune_nemotron.py \
  --train-manifest $TRAIN_MANIFEST \
  --val-manifest $VAL_MANIFEST \
  --base-model $BASE_MODEL \
  --output-nemo /srv/models/finetuned_nemotron_v3_lr1e6_ep3.nemo \
  --freeze-mode decoder_only \
  --max-epochs 3 \
  --batch-size 1 \
  --lr 1e-6 \
  --language en-US \
  --precision bf16-mixed

python3.11 scripts/evaluate_manifest.py \
  --model /srv/models/finetuned_nemotron_v3_lr1e6_ep3.nemo \
  --manifest $TEST_MANIFEST \
  --language en-US \
  --output-jsonl results/hparam_tuning/v3_eval.jsonl

echo "All fine-tuned models:"
ls -lh /srv/models/finetuned_nemotron_v*.nemo

echo "Evaluation files:"
ls -lh results/hparam_tuning
