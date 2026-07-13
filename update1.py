export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_HOME=/usr/local/cuda-12.4
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/nvvm/lib64:$LD_LIBRARY_PATH
unset NUMBA_CUDA_USE_NVIDIA_BINDING
export TOKENIZERS_PARALLELISM=false
python3.11 scripts/prepare_dataset.py --csv data/inspira_transcripts.csv --wav-dir raw_wavs --out-dir data
find data/audio_16k -maxdepth 1 -type f -name "*.wav" -print
ls -lh data/audio_16k
python3.11 scripts/auto_align_chunks_with_base_asr.py --csv data/inspira_transcripts.csv --wav-dir data/audio_16k --base-model /srv/nemotron-3.5-asr-streaming-0.6b.nemo --out-dir data/audio_chunks --manifest data/manifests/aligned_chunk_manifest.json --audit data/manifests/alignment_audit.csv --language en-US --chunk-sec 10 --min-score 0.30
wc -l data/manifests/aligned_chunk_manifest.json
head -n 3 data/manifests/aligned_chunk_manifest.json
python3.11 -c "import csv; rows=list(csv.DictReader(open('data/manifests/alignment_audit.csv',encoding='utf-8'))); bad=[r for r in rows if float(r['match_score'])<0.45]; print('low-score rows:',len(bad)); [print(r['match_score'],r['chunk'],r['aligned_text'][:120]) for r in bad]"
python3.11 scripts/split_by_usecase_manifest.py --input data/manifests/aligned_chunk_manifest.json --out-dir data/manifests --seed 42
wc -l data/manifests/train_aligned_manifest.json data/manifests/val_aligned_manifest.json data/manifests/test_aligned_manifest.json
python3.11 -c "import json; from pathlib import Path; f=lambda p:{json.loads(x).get('use_case') for x in Path(p).read_text().splitlines() if x.strip()}; tr=f('data/manifests/train_aligned_manifest.json'); va=f('data/manifests/val_aligned_manifest.json'); te=f('data/manifests/test_aligned_manifest.json'); print('train:',tr); print('val:',va); print('test:',te); print('overlap:',(tr&va)|(tr&te)|(va&te))"
python3.11 scripts/evaluate_manifest.py --model /srv/nemotron-3.5-asr-streaming-0.6b.nemo --manifest data/manifests/test_aligned_manifest.json --language en-US --output-jsonl results/hparam_tuning/base_eval.jsonl
python3.11 scripts/finetune_nemotron.py --train-manifest data/manifests/train_aligned_manifest.json --val-manifest data/manifests/val_aligned_manifest.json --base-model /srv/nemotron-3.5-asr-streaming-0.6b.nemo --output-nemo /srv/models/finetuned_clean_lr1e6_ep1.nemo --freeze-mode decoder_only --max-epochs 1 --batch-size 1 --accumulate-grad-batches 1 --lr 1e-6 --language en-US --precision bf16-mixed --num-workers 0
python3.11 scripts/evaluate_manifest.py --model /srv/models/finetuned_clean_lr1e6_ep1.nemo --manifest data/manifests/test_aligned_manifest.json --language en-US --output-jsonl results/hparam_tuning/clean_eval.jsonl
python3.11 scripts/augment_train_manifest.py --train-manifest data/manifests/train_aligned_manifest.json --out-manifest data/manifests/train_aligned_aug_manifest.json --out-audio-dir data/audio_aug --keep-original
wc -l data/manifests/train_aligned_manifest.json data/manifests/train_aligned_aug_manifest.json
python3.11 -c "import json,collections; from pathlib import Path; rows=[json.loads(x) for x in Path('data/manifests/train_aligned_aug_manifest.json').read_text().splitlines() if x.strip()]; print(collections.Counter(r.get('augmentation','original') for r in rows))"
python3.11 scripts/finetune_nemotron.py --train-manifest data/manifests/train_aligned_aug_manifest.json --val-manifest data/manifests/val_aligned_manifest.json --base-model /srv/nemotron-3.5-asr-streaming-0.6b.nemo --output-nemo /srv/models/finetuned_robust_lr1e6_ep1.nemo --freeze-mode decoder_only --max-epochs 1 --batch-size 1 --accumulate-grad-batches 2 --lr 1e-6 --language en-US --precision bf16-mixed --num-workers 0
python3.11 scripts/evaluate_manifest.py --model /srv/models/finetuned_robust_lr1e6_ep1.nemo --manifest data/manifests/test_aligned_manifest.json --language en-US --output-jsonl results/hparam_tuning/robust_eval.jsonl
python3.11 -c "import json; from pathlib import Path; fs=['base_eval.jsonl','clean_eval.jsonl','robust_eval.jsonl']; [(lambda rows,n:print(n,'rows=',len(rows),'WER=',round(sum(float(r['wer']) for r in rows)/max(1,len(rows)),2),'CER=',round(sum(float(r['cer']) for r in rows)/max(1,len(rows)),2)))([json.loads(x) for x in Path('results/hparam_tuning/'+f).read_text().splitlines() if x.strip()],f) for f in fs]"
cp /srv/models/finetuned_clean_lr1e6_ep1.nemo /srv/models/finetuned_nemotron_final.nemo
cp /srv/models/finetuned_robust_lr1e6_ep1.nemo /srv/models/finetuned_nemotron_final.nemo
cp /srv/nemotron-3.5-asr-streaming-0.6b.nemo /srv/models/finetuned_nemotron_final.nemo
ls -lh /srv/models/finetuned_nemotron_final.nemo
ls -lh /home/CORP/re_nikitav/nemotron_finetuned/ft_models/finetuned_nemotron_final.nemo
exit
cd /home/CORP/re_nikitav/nemotron_finetuned && docker run --gpus all -it --rm -p 8003:8003 -v "$PWD":/workspace -v "$PWD/ft_models":/srv/models -e MODEL_NAME=/srv/models/finetuned_nemotron_final.nemo -e VAD_START_MARGIN=1.8 -e VAD_MIN_NOISE_RMS=0.002 -e PRE_SPEECH_MS=500 -e NEMO_END_SILENCE_MS=900 -e FINALIZE_PAD_MS=800 -e CONTEXT_RIGHT=2 -e NEMO_MAX_SYMBOLS=15 nemotron_finetuned uvicorn app.main:app --host 0.0.0.0 --port 8003

