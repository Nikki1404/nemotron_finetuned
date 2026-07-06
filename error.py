cd /home/CORP/re_nikitav/nemotron_finetuned && mkdir -p ft_models results/hparam_tuning

cd /home/CORP/re_nikitav/nemotron_finetuned && docker run --gpus all -it --rm -v $PWD:/workspace -v $PWD/ft_models:/srv/models nemotron_finetuned bash

cd /workspace && mkdir -p /srv/models results/hparam_tuning && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && export CUDA_HOME=/usr/local/cuda-12.4 && export LD_LIBRARY_PATH=/usr/local/cuda-12.4/nvvm/lib64:$LD_LIBRARY_PATH && unset NUMBA_CUDA_USE_NVIDIA_BINDING

cd /workspace && python3.11 scripts/augment_train_manifest.py --train-manifest data/manifests/train_aligned_manifest.json --out-manifest data/manifests/train_aligned_aug_manifest.json --out-audio-dir data/audio_aug --keep-original

cd /workspace && python3.11 scripts/finetune_nemotron.py --train-manifest data/manifests/train_aligned_aug_manifest.json --val-manifest data/manifests/val_aligned_manifest.json --base-model /srv/nemotron-3.5-asr-streaming-0.6b.nemo --output-nemo /srv/models/finetuned_nemotron_v1_lr3e6_ep2.nemo --freeze-mode decoder_only --max-epochs 2 --batch-size 1 --lr 3e-6 --language en-US --precision bf16-mixed

cp /srv/models/finetuned_nemotron_v1_lr3e6_ep2.nemo /srv/models/finetuned_nemotron_final.nemo && ls -lh /srv/models

cd /workspace && python3.11 scripts/evaluate_manifest.py --model /srv/models/finetuned_nemotron_final.nemo --manifest data/manifests/test_aligned_manifest.json --language en-US --output-jsonl results/hparam_tuning/final_eval.jsonl

cd /workspace && chmod +x scripts/run_hyparam_tuning.sh && bash scripts/run_hyparam_tuning.sh


cd /workspace && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && export CUDA_HOME=/usr/local/cuda-12.4 && export LD_LIBRARY_PATH=/usr/local/cuda-12.4/nvvm/lib64:$LD_LIBRARY_PATH && unset NUMBA_CUDA_USE_NVIDIA_BINDING && mkdir -p /srv/models results/hparam_tuning && chmod +x scripts/run_hyparam_tuning.sh && bash scripts/run_hyparam_tuning.shs


cd /workspace && python3.11 scripts/compare_models_report.py --base results/hparam_tuning/base_eval.jsonl --v1 results/hparam_tuning/v1_eval.jsonl --v2 results/hparam_tuning/v2_eval.jsonl --v3 results/hparam_tuning/v3_eval.jsonl --out results/hparam_tuning/model_comparison_report.md

cp /srv/models/finetuned_nemotron_v1_lr3e6_ep2.nemo /srv/models/finetuned_nemotron_final.nemo && ls -lh /srv/models/finetuned_nemotron_final.nemo

cd /home/CORP/re_nikitav/nemotron_finetuned && docker run --gpus all -it --rm -p 8003:8003 -v $PWD:/workspace -v $PWD/ft_models:/srv/models -e MODEL_NAME=/srv/models/finetuned_nemotron_final.nemo nemotron_finetuned uvicorn app.main:app --host 0.0.0.0 --port 8003

ps -ef | grep -E "uvicorn|app.main" | grep -v grep
python3.11 -c "import app.main,inspect; print(inspect.getfile(app.main))"
grep -n "v1/audio/transcriptions\|realtime-custom-vad\|AUDIO_LOG_DIR" /workspace/app/main.py
curl -s http://localhost:8003/


(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav# docker exec -it bcc6c2abcf6c bash
root@bcc6c2abcf6c:/srv# ps -ef | grep -E "uvicorn|app.main" | grep -v grep
root           1       0 18 16:15 pts/0    00:00:45 /usr/local/bin/python3.11 /usr/local/bin/uvicorn app.main:app --host 0.0.0.0 --port 8003
root@bcc6c2abcf6c:/srv# python3.11 -c "import app.main,inspect; print(inspect.getfile(app.main))"
DEBUG: Startup cfg.model_name='/srv/models/finetuned_nemotron_final.nemo' cfg.asr_backend='nemotron'
/srv/app/main.py
root@bcc6c2abcf6c:/srv# grep -n "v1/audio/transcriptions\|realtime-custom-vad\|AUDIO_LOG_DIR" /workspace/app/main.py
41:AUDIO_LOG_DIR = Path(os.getenv("AUDIO_LOG_DIR", "/srv/audio_logs"))
42:AUDIO_LOG_DIR.mkdir(parents=True, exist_ok=True)
176:        "audio_log_dir": str(AUDIO_LOG_DIR),
185:        "websocket_endpoint": "/asr/realtime-custom-vad",
186:        "openai_transcription_endpoint": "/v1/audio/transcriptions",
187:        "audio_log_dir": str(AUDIO_LOG_DIR),
259:    Used by /v1/audio/transcriptions.
264:    session_dir = AUDIO_LOG_DIR / session_id
431:@app.post("/v1/audio/transcriptions")
533:@app.websocket("/asr/realtime-custom-vad")
540:    session_dir = AUDIO_LOG_DIR / session_id
