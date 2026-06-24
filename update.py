mkdir -p ft_models

docker run --gpus all -it --rm -v $PWD:/workspace -v $PWD/ft_models:/srv/models nemotron-asr bash


cd /workspace
ls
python3.11 scripts/prepare_dataset.py --csv data/inspira_transcripts.csv --wav-dir raw_wavs --out-dir data

ls data/manifests

python3.11 scripts/evaluate_manifest.py --model /srv/nemotron-3.5-asr-streaming-0.6b.nemo --manifest data/manifests/test_manifest.json --language en-US --out-jsonl results_base.jsonl

python3.11 scripts/finetune_nemotron.py --train-manifest data/manifests/train_manifest.json --val-manifest data/manifests/val_manifest.json --base-model /srv/nemotron-3.5-asr-streaming-0.6b.nemo --output-nemo /srv/models/nemotron_inspira_decoder_ft.nemo --freeze-mode decoder_only --max-epochs 5 --batch-size 1 --lr 5e-6 --language en-US

python3.11 scripts/evaluate_manifest.py --model /srv/models/nemotron_inspira_decoder_ft.nemo --manifest data/manifests/test_manifest.json --language en-US --out-jsonl results_finetuned.jsonl

exit

ls -lh ft_models/

docker run --gpus all -it --rm -p 8003:8003 -v $PWD/ft_models:/srv/models -e MODEL_NAME=/srv/models/nemotron_inspira_decoder_ft.nemo nemotron-asr
