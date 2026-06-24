root@c6f79d6e94db:/workspace# python3.11 scripts/evaluate_manifest.py --model /srv/nemotron-3.5-asr-streaming-0.6b.nemo --manifest data/manifests/test_manifest.json --language en-US --output-jsonl results_base.jsonl
OneLogger: Setting error_handling_strategy to DISABLE_QUIETLY_AND_REPORT_METRIC_ERROR for rank (rank=0) with OneLogger disabled. To override: explicitly set error_handling_strategy parameter.
No exporters were provided. This means that no telemetry data will be collected.
[NeMo I 2026-06-24 20:13:56 mixins:194] Tokenizer SentencePieceTokenizer initialized with 13087 tokens
[NeMo W 2026-06-24 20:14:01 modelPT:287] You tried to register an artifact under config key=tokenizer.model_path but an artifact for it has already been registered.
[NeMo W 2026-06-24 20:14:01 modelPT:287] You tried to register an artifact under config key=tokenizer.vocab_path but an artifact for it has already been registered.
[NeMo I 2026-06-24 20:14:02 mixins:194] Tokenizer SentencePieceTokenizer initialized with 13087 tokens
[NeMo W 2026-06-24 20:14:08 modelPT:175] If you intend to do training or fine-tuning, please call the ModelPT.setup_training_data() method and provide a valid configuration file to setup the train data loader.
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

[NeMo W 2026-06-24 20:14:08 modelPT:182] If you intend to do validation, please call the ModelPT.setup_validation_data() or ModelPT.setup_multiple_validation_data() method and provide a valid configuration file to setup the validation data loader(s).
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

[NeMo W 2026-06-24 20:14:08 modelPT:189] Please call the ModelPT.setup_test_data() or ModelPT.setup_multiple_test_data() method and provide a valid configuration file to setup the test data loader(s).
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

[NeMo I 2026-06-24 20:14:15 rnnt_models:226] Using RNNT Loss : warprnnt_numba
    Loss warprnnt_numba_kwargs: {'fastemit_lambda': 0.005, 'clamp': -1.0}
[NeMo I 2026-06-24 20:14:15 rnnt_models:226] Using RNNT Loss : warprnnt_numba
    Loss warprnnt_numba_kwargs: {'fastemit_lambda': 0.005, 'clamp': -1.0}
[NeMo I 2026-06-24 20:14:15 rnnt_models:226] Using RNNT Loss : warprnnt_numba
    Loss warprnnt_numba_kwargs: {'fastemit_lambda': 0.005, 'clamp': -1.0}
[NeMo I 2026-06-24 20:14:15 rnnt_bpe_models_prompt:146] Model with prompt feature has been initialized (RNNT-only)
[NeMo I 2026-06-24 20:14:18 save_restore_connector:287] Model EncDecRNNTBPEModelWithPrompt was successfully restored from /srv/nemotron-3.5-asr-streaming-0.6b.nemo.
[NeMo I 2026-06-24 20:14:19 mixins:969] Inference prompt set to 'en-US' (index 0)
[eval] Loaded model: /srv/nemotron-3.5-asr-streaming-0.6b.nemo
[eval] Files: 1
[NeMo W 2026-06-24 20:14:20 dataloader:881] The following configuration keys are ignored by Lhotse dataloader: window_stride,trim_silence,subsampling_factor,prompt_dictionary,initialize_prompt_feature,num_prompts,default_lang,labels
[NeMo W 2026-06-24 20:14:20 dataloader:533] You are using a non-tarred dataset and requested tokenization during data sampling (pretokenize=True). This will cause the tokenization to happen in the main (GPU) process,possibly impacting the training speed if your tokenizer is very large.If the impact is noticable, set pretokenize=False in dataloader config.(note: that will disable token-per-second filtering and 2D bucketing features)
Transcribing: 0it [00:00, ?it/s]
Traceback (most recent call last):
  File "/workspace/scripts/evaluate_manifest.py", line 136, in <module>
    main()
  File "/workspace/scripts/evaluate_manifest.py", line 96, in main
    hyps = model.transcribe(audio_files, batch_size=args.batch_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/torch/utils/_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/nemo/collections/asr/models/rnnt_bpe_models_prompt.py", line 629, in transcribe
    return super().transcribe(
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/torch/utils/_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/nemo/collections/asr/models/rnnt_models.py", line 317, in transcribe
    return super().transcribe(
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/torch/utils/_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/nemo/collections/asr/parts/mixins/transcription.py", line 298, in transcribe
    for processed_outputs in generator:
  File "/usr/local/lib/python3.11/site-packages/nemo/collections/asr/parts/mixins/transcription.py", line 393, in transcribe_generator
    for test_batch in tqdm(dataloader, desc="Transcribing", disable=not verbose):
  File "/usr/local/lib/python3.11/site-packages/tqdm/std.py", line 1181, in __iter__
    for obj in iterable:
  File "/usr/local/lib/python3.11/site-packages/torch/utils/data/dataloader.py", line 708, in __next__
    data = self._next_data()
           ^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/torch/utils/data/dataloader.py", line 764, in _next_data
    data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/torch/utils/data/_utils/fetch.py", line 54, in fetch
    data = self.dataset[possibly_batched_index]
           ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/nemo/collections/asr/data/audio_to_text_lhotse_prompt_index.py", line 142, in __getitem__
    prompt_indices = torch.tensor([self._get_prompt_index_for_cut(c) for c in cuts], dtype=torch.long)
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/nemo/collections/asr/data/audio_to_text_lhotse_prompt_index.py", line 142, in <listcomp>
    prompt_indices = torch.tensor([self._get_prompt_index_for_cut(c) for c in cuts], dtype=torch.long)
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/nemo/collections/asr/data/audio_to_text_lhotse_prompt_index.py", line 130, in _get_prompt_index_for_cut
    return self._get_prompt_index(cut.supervisions[0].language)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/nemo/collections/asr/data/audio_to_text_lhotse_prompt_index.py", line 96, in _get_prompt_index
    raise ValueError(
ValueError: Unknown prompt key: 'None'. Available: ['en-US', 'en', 'en-GB', 'enGB', 'es-ES', 'esES', 'es-US', 'es', 'zh-CN', 'zh-ZH']...
