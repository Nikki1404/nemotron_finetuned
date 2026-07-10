(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav/nemotron_finetuned_updated# docker run --gpus all -it --rm -p 8003:8003 -v $PWD:/workspace -v $PWD/ft_models:/srv/models -e MODEL_NAME=/srv/models/finetuned_nemotron_final.nemo nemotron_finetuned uvicorn app.main:app --host 0.0.0.0 --port 8003

==========
== CUDA ==
==========

CUDA Version 12.4.1

Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.

DEBUG: Startup cfg.model_name='/srv/models/finetuned_nemotron_final.nemo' cfg.asr_backend='nemotron' sample_rate=16000 vad_frame_ms=20 vad_start_margin=1.5 vad_min_noise_rms=0.0015 pre_speech_ms=700 max_utt_ms=30000
INFO:     Started server process [1]
INFO:     Waiting for application startup.
2026-07-10 08:20:18,224 | INFO | asr_server | Server startup initiated
2026-07-10 08:20:18,225 | INFO | asr_server | Preloading ASR engines...
2026-07-10 08:20:18,225 | INFO | asr_server | Initializing engine: nemotron (/srv/models/finetuned_nemotron_final.nemo)
2026-07-10 08:20:22,951 | WARNING | nv_one_logger.api.config | OneLogger: Setting error_handling_strategy to DISABLE_QUIETLY_AND_REPORT_METRIC_ERROR for rank (rank=0) with OneLogger disabled. To override: explicitly set error_handling_strategy parameter.
2026-07-10 08:20:22,961 | INFO | nv_one_logger.exporter.export_config_manager | Final configuration contains 0 exporter(s)
2026-07-10 08:20:22,961 | WARNING | nv_one_logger.training_telemetry.api.training_telemetry_provider | No exporters were provided. This means that no telemetry data will be collected.
2026-07-10 08:20:25,031 | ERROR | asr_server | Failed to preload 'nemotron'
Traceback (most recent call last):
  File "/srv/app/main.py", line 142, in preload_engines
    load_sec = engine.load()
               ^^^^^^^^^^^^^
  File "/srv/app/asr_engines/nemotron_asr.py", line 125, in load
    self.model = nemo_asr.models.EncDecRNNTBPEModelWithPrompt.restore_from(
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/nemo/collections/asr/models/rnnt_bpe_models_prompt.py", line 132, in restore_from
    return EncDecRNNTBPEModel.restore_from(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/nemo/core/classes/modelPT.py", line 483, in restore_from
    raise FileNotFoundError(f"Can't find {restore_path}")
FileNotFoundError: Can't find /srv/models/finetuned_nemotron_final.nemo
