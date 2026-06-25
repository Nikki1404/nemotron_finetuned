NUMBA_CUDA_USE_NVIDIA_BINDING=1 \
LD_LIBRARY_PATH=/usr/local/cuda-12.4/nvvm/lib64:$LD_LIBRARY_PATH \
python3.11 scripts/finetune_nemotron.py \
  --train-manifest data/manifests/train_manifest.json \
  --val-manifest data/manifests/val_manifest.json \
  --base-model /srv/nemotron-3.5-asr-streaming-0.6b.nemo \
  --output-nemo /srv/models/nemotron_inspira_decoder_ft.nemo \
  --freeze-mode decoder_only \
  --max-epochs 5 \
  --batch-size 1 \
  --lr 5e-6 \
  --language en-US \
  --precision bf16-mixed
