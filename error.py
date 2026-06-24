python3.11 - <<'PY'
import nemo.collections.asr as nemo_asr

m = nemo_asr.models.ASRModel.restore_from(
    "/srv/nemotron-3.5-asr-streaming-0.6b.nemo"
)

print(type(m))
print("set_default_prompt:", hasattr(m, "set_default_prompt"))
print("set_inference_prompt:", hasattr(m, "set_inference_prompt"))
PY



python3.11 scripts/evaluate_manifest.py --model /srv/nemotron-3.5-asr-streaming-0.6b.nemo --manifest data/manifests/test_manifest.json --language en-US --output-jsonl results_base.jsonl
