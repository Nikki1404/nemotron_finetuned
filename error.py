root@c6f79d6e94db:/workspace# python3.11 scripts/finetune_nemotron.py --train-manifest data/manifests/train_manifest.json --val-manifest data/manifests/val_manifest.json --base-model /srv/nemotron-3.5-asr-streaming-0.6b.nemo --output-nemo /srv/models/nemotron_inspira_decoder_ft.nemo --freeze-mode decoder_only --max-epochs 5 --batch-size 1 --lr 5e-6 --language en-US
OneLogger: Setting error_handling_strategy to DISABLE_QUIETLY_AND_REPORT_METRIC_ERROR for rank (rank=0) with OneLogger disabled. To override: explicitly set error_handling_strategy parameter.
No exporters were provided. This means that no telemetry data will be collected.
[NeMo I 2026-06-24 21:41:01 mixins:194] Tokenizer SentencePieceTokenizer initialized with 13087 tokens
[NeMo W 2026-06-24 21:41:07 modelPT:287] You tried to register an artifact under config key=tokenizer.model_path but an artifact for it has already been registered.
[NeMo W 2026-06-24 21:41:07 modelPT:287] You tried to register an artifact under config key=tokenizer.vocab_path but an artifact for it has already been registered.
[NeMo I 2026-06-24 21:41:07 mixins:194] Tokenizer SentencePieceTokenizer initialized with 13087 tokens
[NeMo W 2026-06-24 21:41:13 modelPT:175] If you intend to do training or fine-tuning, please call the ModelPT.setup_training_data() method and provide a valid configuration file to setup the train data loader.
    Train config :
    manifest_filepath: null
    sample_rate: 16000
    use_lhotse: true
    shard_manifests: true
    batch_duration: 200
    quadratic_duration: 15
    num_buckets: 30
    shuffle: true
    num_workers: 8
    pin_memory: true
    max_duration: 20
    min_duration: 0.1
    is_tarred: true
    tarred_audio_filepaths: null
    shuffle_n: 2048
    slice_length: 100
    bucketing_strategy: fully_randomized
    bucketing_batch_size: null
    bucket_buffer_size: 10000
    shuffle_buffer_size: 10000
    prompt_field: target_lang
    prompt_dictionary:
      en-US: 0
      en: 0
      en-GB: 1
      enGB: 1
      es-ES: 2
      esES: 2
      es-US: 3
      es: 3
      zh-CN: 4
      zh-ZH: 4
      zh-TW: 5
      hi-IN: 6
      hi: 6
      hi-HI: 6
      ar-AR: 7
      ar: 7
      fr-FR: 8
      fr: 8
      de-DE: 9
      de: 9
      ja-JP: 10
      ja-JA: 10
      ru-RU: 11
      ru: 11
      pt-BR: 12
      pt-PT: 13
      pt: 13
      ko-KR: 14
      ko: 14
      ko-KO: 14
      it-IT: 15
      it: 15
      nl-NL: 16
      nl: 16
      pl-PL: 17
      pl: 17
      tr-TR: 18
      tr: 18
      uk-UA: 19
      uk: 19
      ro-RO: 20
      ro: 20
      el-GR: 21
      el: 21
      cs-CZ: 22
      cs: 22
      hu-HU: 23
      hu: 23
      sv-SE: 24
      sv: 24
      da-DK: 25
      da: 25
      fi-FI: 26
      fi: 26
      no-NO: 27
      'no': 27
      nb-NO: 103
      nb: 103
      nn-NO: 104
      nn: 104
      sk-SK: 28
      sk: 28
      hr-HR: 29
      hr: 29
      bg-BG: 30
      bg: 30
      lt-LT: 31
      lt: 31
      et-EE: 60
      et: 60
      lv-LV: 61
      lv: 61
      sl-SI: 62
      sl: 62
      th-TH: 32
      vi-VN: 33
      id-ID: 34
      ms-MY: 35
      bn-IN: 36
      ur-PK: 37
      fa-IR: 38
      ta-IN: 39
      te-IN: 40
      mr-IN: 41
      gu-IN: 42
      kn-IN: 43
      ml-IN: 44
      si-LK: 45
      ne-NP: 46
      km-KH: 47
      sw-KE: 48
      am-ET: 49
      ha-NG: 50
      zu-ZA: 51
      yo-NG: 52
      ig-NG: 53
      af-ZA: 54
      rw-RW: 55
      so-SO: 56
      ny-MW: 57
      ln-CD: 58
      or-KE: 59
      he-IL: 64
      ku-TR: 65
      az-AZ: 66
      ka-GE: 67
      hy-AM: 68
      uz-UZ: 69
      tg-TJ: 70
      ky-KG: 71
      qu-PE: 80
      ay-BO: 81
      gn-PY: 82
      nah-MX: 83
      mi-NZ: 96
      haw-US: 97
      sm-WS: 98
      to-TO: 99
      fr-CA: 100
      mt-MT: 102
      auto: 101
    num_prompts: 128
    subsampling_factor: 8
    lang_field: target_lang
    training_mode: true
    input_cfg: null
    use_bucketing: true
    defer_setup: true

[NeMo W 2026-06-24 21:41:13 modelPT:182] If you intend to do validation, please call the ModelPT.setup_validation_data() or ModelPT.setup_multiple_validation_data() method and provide a valid configuration file to setup the validation data loader(s).
    Validation config :
    manifest_filepath: null
    sample_rate: 16000
    batch_size: 2
    shuffle: false
    use_start_end_token: false
    num_workers: 2
    pin_memory: true
    batch_duration: null
    use_lhotse: true
    use_bucketing: false
    max_cuts: 8
    prompt_field: target_lang
    prompt_dictionary:
      en-US: 0
      en: 0
      en-GB: 1
      enGB: 1
      es-ES: 2
      esES: 2
      es-US: 3
      es: 3
      zh-CN: 4
      zh-ZH: 4
      zh-TW: 5
      hi-IN: 6
      hi: 6
      hi-HI: 6
      ar-AR: 7
      ar: 7
      fr-FR: 8
      fr: 8
      de-DE: 9
      de: 9
      ja-JP: 10
      ja-JA: 10
      ru-RU: 11
      ru: 11
      pt-BR: 12
      pt-PT: 13
      pt: 13
      ko-KR: 14
      ko: 14
      ko-KO: 14
      it-IT: 15
      it: 15
      nl-NL: 16
      nl: 16
      pl-PL: 17
      pl: 17
      tr-TR: 18
      tr: 18
      uk-UA: 19
      uk: 19
      ro-RO: 20
      ro: 20
      el-GR: 21
      el: 21
      cs-CZ: 22
      cs: 22
      hu-HU: 23
      hu: 23
      sv-SE: 24
      sv: 24
      da-DK: 25
      da: 25
      fi-FI: 26
      fi: 26
      no-NO: 27
      'no': 27
      nb-NO: 103
      nb: 103
      nn-NO: 104
      nn: 104
      sk-SK: 28
      sk: 28
      hr-HR: 29
      hr: 29
      bg-BG: 30
      bg: 30
      lt-LT: 31
      lt: 31
      et-EE: 60
      et: 60
      lv-LV: 61
      lv: 61
      sl-SI: 62
      sl: 62
      th-TH: 32
      vi-VN: 33
      id-ID: 34
      ms-MY: 35
      bn-IN: 36
      ur-PK: 37
      fa-IR: 38
      ta-IN: 39
      te-IN: 40
      mr-IN: 41
      gu-IN: 42
      kn-IN: 43
      ml-IN: 44
      si-LK: 45
      ne-NP: 46
      km-KH: 47
      sw-KE: 48
      am-ET: 49
      ha-NG: 50
      zu-ZA: 51
      yo-NG: 52
      ig-NG: 53
      af-ZA: 54
      rw-RW: 55
      so-SO: 56
      ny-MW: 57
      ln-CD: 58
      or-KE: 59
      he-IL: 64
      ku-TR: 65
      az-AZ: 66
      ka-GE: 67
      hy-AM: 68
      uz-UZ: 69
      tg-TJ: 70
      ky-KG: 71
      qu-PE: 80
      ay-BO: 81
      gn-PY: 82
      nah-MX: 83
      mi-NZ: 96
      haw-US: 97
      sm-WS: 98
      to-TO: 99
      fr-CA: 100
      mt-MT: 102
      auto: 101
    num_prompts: 128
    subsampling_factor: 8
    training_mode: true
    name: null

[NeMo W 2026-06-24 21:41:13 modelPT:189] Please call the ModelPT.setup_test_data() or ModelPT.setup_multiple_test_data() method and provide a valid configuration file to setup the test data loader(s).
    Test config :
    manifest_filepath: null
    sample_rate: 16000
    batch_size: 16
    shuffle: false
    use_start_end_token: false
    num_workers: 8
    pin_memory: true
    use_lhotse: true
    use_bucketing: false
    prompt_field: target_lang
    prompt_dictionary:
      en-US: 0
      en: 0
      en-GB: 1
      enGB: 1
      es-ES: 2
      esES: 2
      es-US: 3
      es: 3
      zh-CN: 4
      zh-ZH: 4
      zh-TW: 5
      hi-IN: 6
      hi: 6
      hi-HI: 6
      ar-AR: 7
      ar: 7
      fr-FR: 8
      fr: 8
      de-DE: 9
      de: 9
      ja-JP: 10
      ja-JA: 10
      ru-RU: 11
      ru: 11
      pt-BR: 12
      pt-PT: 13
      pt: 13
      ko-KR: 14
      ko: 14
      ko-KO: 14
      it-IT: 15
      it: 15
      nl-NL: 16
      nl: 16
      pl-PL: 17
      pl: 17
      tr-TR: 18
      tr: 18
      uk-UA: 19
      uk: 19
      ro-RO: 20
      ro: 20
      el-GR: 21
      el: 21
      cs-CZ: 22
      cs: 22
      hu-HU: 23
      hu: 23
      sv-SE: 24
      sv: 24
      da-DK: 25
      da: 25
      fi-FI: 26
      fi: 26
      no-NO: 27
      'no': 27
      nb-NO: 103
      nb: 103
      nn-NO: 104
      nn: 104
      sk-SK: 28
      sk: 28
      hr-HR: 29
      hr: 29
      bg-BG: 30
      bg: 30
      lt-LT: 31
      lt: 31
      et-EE: 60
      et: 60
      lv-LV: 61
      lv: 61
      sl-SI: 62
      sl: 62
      th-TH: 32
      vi-VN: 33
      id-ID: 34
      ms-MY: 35
      bn-IN: 36
      ur-PK: 37
      fa-IR: 38
      ta-IN: 39
      te-IN: 40
      mr-IN: 41
      gu-IN: 42
      kn-IN: 43
      ml-IN: 44
      si-LK: 45
      ne-NP: 46
      km-KH: 47
      sw-KE: 48
      am-ET: 49
      ha-NG: 50
      zu-ZA: 51
      yo-NG: 52
      ig-NG: 53
      af-ZA: 54
      rw-RW: 55
      so-SO: 56
      ny-MW: 57
      ln-CD: 58
      or-KE: 59
      he-IL: 64
      ku-TR: 65
      az-AZ: 66
      ka-GE: 67
      hy-AM: 68
      uz-UZ: 69
      tg-TJ: 70
      ky-KG: 71
      qu-PE: 80
      ay-BO: 81
      gn-PY: 82
      nah-MX: 83
      mi-NZ: 96
      haw-US: 97
      sm-WS: 98
      to-TO: 99
      fr-CA: 100
      mt-MT: 102
      auto: 101
    num_prompts: 128
    subsampling_factor: 8
    training_mode: false

[NeMo I 2026-06-24 21:41:21 rnnt_models:226] Using RNNT Loss : warprnnt_numba
    Loss warprnnt_numba_kwargs: {'fastemit_lambda': 0.005, 'clamp': -1.0}
[NeMo I 2026-06-24 21:41:21 rnnt_models:226] Using RNNT Loss : warprnnt_numba
    Loss warprnnt_numba_kwargs: {'fastemit_lambda': 0.005, 'clamp': -1.0}
[NeMo I 2026-06-24 21:41:21 rnnt_models:226] Using RNNT Loss : warprnnt_numba
    Loss warprnnt_numba_kwargs: {'fastemit_lambda': 0.005, 'clamp': -1.0}
[NeMo I 2026-06-24 21:41:21 rnnt_bpe_models_prompt:146] Model with prompt feature has been initialized (RNNT-only)
[NeMo I 2026-06-24 21:41:23 save_restore_connector:287] Model EncDecRNNTBPEModelWithPrompt was successfully restored from /srv/nemotron-3.5-asr-streaming-0.6b.nemo.
[NeMo I 2026-06-24 21:41:23 mixins:969] Inference prompt set to 'en-US' (index 0)
[params] total=637,997,088 trainable=24,395,808 (3.82%)
[NeMo I 2026-06-24 21:41:24 collections:201] Dataset loaded with 4 files totalling 0.08 hours
[NeMo I 2026-06-24 21:41:24 collections:202] 1 files were filtered totalling 0.05 hours
[NeMo I 2026-06-24 21:41:25 collections:201] Dataset loaded with 1 files totalling 0.01 hours
[NeMo I 2026-06-24 21:41:25 collections:202] 0 files were filtered totalling 0.00 hours
Using bfloat16 Automatic Mixed Precision (AMP)
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
[train] Starting fine-tuning...
You are using a CUDA device ('NVIDIA A10G') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
[NeMo I 2026-06-24 21:41:28 modelPT:781] Optimizer config = AdamW (
    Parameter Group 0
        amsgrad: False
        betas: [0.9, 0.98]
        capturable: False
        differentiable: False
        eps: 1e-08
        foreach: None
        fused: None
        lr: 5e-06
        maximize: False
        weight_decay: 0.001
    )
[NeMo I 2026-06-24 21:41:28 lr_scheduler:995] Scheduler "<nemo.core.optim.lr_scheduler.CosineAnnealing object at 0x71230aaa0b50>"
    will be used during training (effective maximum steps = 20) -
    Parameters :
    (warmup_steps: 5
    min_lr: 2.5000000000000004e-07
    max_steps: 20
    )

  | Name              | Type                              | Params | Mode
--------------------------------------------------------------------------------
0 | preprocessor      | AudioToMelSpectrogramPreprocessor | 0      | train
1 | encoder           | ConformerEncoder                  | 609 M  | train
2 | decoder           | RNNTDecoder                       | 14.9 M | train
3 | joint             | RNNTJoint                         | 9.5 M  | train
4 | loss              | RNNTLoss                          | 0      | train
5 | spec_augmentation | SpectrogramAugmentation           | 0      | train
6 | wer               | WER                               | 0      | train
7 | prompt_kernel     | Sequential                        | 4.5 M  | train
--------------------------------------------------------------------------------
24.4 M    Trainable params
613 M     Non-trainable params
637 M     Total params
2,551.988 Total estimated model params size (MB)
710       Modules in train mode
0         Modules in eval mode
Sanity Checking: |                                                                                            | 0/? [00:00<?, ?it/s][NeMo I 2026-06-24 21:41:31 asr_model:198] CUDA graphs disabled for EncDecRNNTBPEModelWithPrompt::RNNTBPEDecoding::GreedyBatchedRNNTInfer
[NeMo I 2026-06-24 21:41:31 asr_model:185] CUDA graphs enabled for EncDecRNNTBPEModelWithPrompt::RNNTBPEDecoding::GreedyBatchedRNNTInfer
Sanity Checking DataLoader 0:   0%|                                                                           | 0/1 [00:00<?, ?it/s]Traceback (most recent call last):
  File "/workspace/scripts/finetune_nemotron.py", line 164, in <module>
    main()
  File "/workspace/scripts/finetune_nemotron.py", line 153, in main
    trainer.fit(model)
  File "/usr/local/lib/python3.11/site-packages/lightning/pytorch/trainer/trainer.py", line 538, in fit
    call._call_and_handle_interrupt(
  File "/usr/local/lib/python3.11/site-packages/lightning/pytorch/trainer/call.py", line 47, in _call_and_handle_interrupt
    return trainer_fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/lightning/pytorch/trainer/trainer.py", line 574, in _fit_impl
    self._run(model, ckpt_path=ckpt_path)
  File "/usr/local/lib/python3.11/site-packages/lightning/pytorch/trainer/trainer.py", line 981, in _run
    results = self._run_stage()
              ^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/lightning/pytorch/trainer/trainer.py", line 1023, in _run_stage
    self._run_sanity_check()
  File "/usr/local/lib/python3.11/site-packages/lightning/pytorch/trainer/trainer.py", line 1052, in _run_sanity_check
    val_loop.run()
  File "/usr/local/lib/python3.11/site-packages/lightning/pytorch/loops/utilities.py", line 178, in _decorator
    return loop_run(self, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/lightning/pytorch/loops/evaluation_loop.py", line 135, in run
    self._evaluation_step(batch, batch_idx, dataloader_idx, dataloader_iter)
  File "/usr/local/lib/python3.11/site-packages/lightning/pytorch/loops/evaluation_loop.py", line 396, in _evaluation_step
    output = call._call_strategy_hook(trainer, hook_name, *step_args)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/lightning/pytorch/trainer/call.py", line 319, in _call_strategy_hook
    output = fn(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/lightning/pytorch/strategies/strategy.py", line 411, in validation_step
    return self.lightning_module.validation_step(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/nemo/collections/asr/models/rnnt_bpe_models_prompt.py", line 512, in validation_step
    tensorboard_logs = self.validation_pass(batch, batch_idx, dataloader_idx)
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/nemo/collections/asr/models/rnnt_bpe_models_prompt.py", line 448, in validation_pass
    signal, signal_len, transcript, transcript_len, prompt_indices = batch
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: not enough values to unpack (expected 5, got 4)
