python3.11 scripts/evaluate_manifest.py --model /srv/nemotron-3.5-asr-streaming-0.6b.nemo --manifest data/manifests/test_manifest.json --language en-US --output-jsonl results_base_full.jsonl

python3.11 scripts/evaluate_manifest.py --model /srv/models/nemotron_inspira_proper_ft.nemo --manifest data/manifests/test_manifest.json --language en-US --output-jsonl results_proper_ft_full.jsonl

exit

python3.11 scripts/evaluate_manifest.py --model /srv/models/nemotron_inspira_proper_ft.nemo --manifest data/manifests/test_manifest.json --language en-US --output-jsonl results_proper_ft_full.jsonl

python3.11 scripts/evaluate_manifest.py --model /srv/models/nemotron_inspira_proper_ft.nemo --manifest data/manifests/test_manifest.json --language en-US --output-jsonl results_proper_ft_full.jsonl
