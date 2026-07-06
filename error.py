root@cx-apps:/home/nikita_verma2# curl -X POST "https://nemotron-finetuned-150916788856.us-central1.run.app/v1/audio/transcriptions" -F "file=@a.wav" -F "model=
nemotron-3.5-asr-streaming-0.6b" -F "language=auto"
{"detail":"Not Found"}root@cx-apps:/home/nikita_verma2# 


(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav/nemotron_finetuned/ft_models# cd ..
(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav/nemotron_finetuned# ls
01_enter_training_container.sh                      README.md                         ft_models         results_base.jsonl
02_prepare_baseline_train_eval_inside_container.sh  README_FINE_TUNING_SAME_IMAGE.md  lightning_logs    results_base_full.jsonl
03_run_finetuned_server.sh                          app                               raw_wavs          results_proper_ft_full.jsonl
04_run_base_server.sh                               client.py                         requirements.txt  scripts
Dockerfile                                          data                              results
(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav/nemotron_finetuned# cd scripts/
(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav/nemotron_finetuned/scripts# ls
augment_train_manifest.py           compare_models_report.py  finetune_nemotron.py  run_hyparam_tuning.sh
auto_align_chunks_with_base_asr.py  evaluate_manifest.py      prepare_dataset.py    split_aligned_manifest.py
