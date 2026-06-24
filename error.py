
[NeMo I 2026-06-24 20:21:14 rnnt_models:226] Using RNNT Loss : warprnnt_numba
    Loss warprnnt_numba_kwargs: {'fastemit_lambda': 0.005, 'clamp': -1.0}
[NeMo I 2026-06-24 20:21:14 rnnt_models:226] Using RNNT Loss : warprnnt_numba
    Loss warprnnt_numba_kwargs: {'fastemit_lambda': 0.005, 'clamp': -1.0}
[NeMo I 2026-06-24 20:21:14 rnnt_models:226] Using RNNT Loss : warprnnt_numba
    Loss warprnnt_numba_kwargs: {'fastemit_lambda': 0.005, 'clamp': -1.0}
[NeMo I 2026-06-24 20:21:14 rnnt_bpe_models_prompt:146] Model with prompt feature has been initialized (RNNT-only)
[NeMo I 2026-06-24 20:21:17 save_restore_connector:287] Model EncDecRNNTBPEModelWithPrompt was successfully restored from /srv/nemotron-3.5-asr-streaming-0.6b.nemo.
[NeMo I 2026-06-24 20:21:17 mixins:969] Inference prompt set to 'en-US' (index 0)
[eval] Loaded model: /srv/nemotron-3.5-asr-streaming-0.6b.nemo
[eval] Files: 1
[NeMo W 2026-06-24 20:21:18 dataloader:881] The following configuration keys are ignored by Lhotse dataloader: trim_silence,prompt_dictionary,initialize_prompt_feature,window_stride,num_prompts,labels,default_lang,subsampling_factor
[NeMo W 2026-06-24 20:21:18 dataloader:533] You are using a non-tarred dataset and requested tokenization during data sampling (pretokenize=True). This will cause the tokenization to happen in the main (GPU) process,possibly impacting the training speed if your tokenizer is very large.If the impact is noticable, set pretokenize=False in dataloader config.(note: that will disable token-per-second filtering and 2D bucketing features)
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
