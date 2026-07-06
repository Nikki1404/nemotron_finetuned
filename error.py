curl -X POST "https://nemotron-finetuned-150916788856.us-central1.run.app/v1/audio/transcriptions" -F "file=@C:\Users\re_nikitav\Downloads\a.wav" -F "model=nemotron-3.5-asr-streaming-0.6b" -F "language=auto"


have a nice day
[NeMo I 2026-07-06 15:05:17 wer:320] WER predicted:Okay, thank you. <en-US> Is there anything else I can help you with today? <en-US> No thanks. <en-US> Thank you have a nice day. <en-US>
                                                                                                                                      [NeMo I 2026-07-06 15:05:17 wer:318] ██████████████████████████████████████████████▊                     | 5/7 [00:02<00:01,  1.98it/s]

[NeMo I 2026-07-06 15:05:17 wer:319] WER reference:is there anything else i can help you with today no thanks thank you for calling inspira financial have a nice day
[NeMo I 2026-07-06 15:05:17 wer:320] WER predicted:Is there anything else I can help you with today? <en-US> No thanks thank you for calling Inspira Financial Have a nice day. <en-US>
                                                                                                                                      [NeMo I 2026-07-06 15:05:17 wer:318] █████████████████████████████████████████████████████████▍          | 6/7 [00:02<00:00,  2.31it/s]

[NeMo I 2026-07-06 15:05:17 wer:319] WER reference:hi hello yeah i am calling because i lost my inspira debit card yesterday evening somewhere around maybe six thirty or seven pm and i am not able to find it anywhere so
[NeMo I 2026-07-06 15:05:17 wer:320] WER predicted:Hi hello yeah i am calling because i lost my inspired debit card yesterday evening somewhere around maybe six thirty or seven pm and i am not able to find it anywhere
                                                                                                                                      [NeMo I 2026-07-06 15:05:17 asr_model:198] CUDA graphs disabled for EncDecRNNTBPEModelWithPrompt::RNNTBPEDecoding::GreedyBatchedRNNTInfer
Epoch 1: 100%|█████████████████████████████████████████████████████████████████████████████| 260/260 [00:58<00:00,  4.44it/s, v_num=9][NeMo I 2026-07-06 15:05:17 asr_model:185] CUDA graphs enabled for EncDecRNNTBPEModelWithPrompt::RNNTBPEDecoding::GreedyBatchedRNNTInfer
`Trainer.fit` stopped: `max_epochs=2` reached.
Epoch 1: 100%|█████████████████████████████████████████████████████████████████████████████| 260/260 [00:58<00:00,  4.44it/s, v_num=9]
Traceback (most recent call last):
  File "/workspace/scripts/finetune_nemotron.py", line 213, in <module>
    main()
  File "/workspace/scripts/finetune_nemotron.py", line 207, in main
    model.save_to(str(output))
  File "/usr/local/lib/python3.11/site-packages/nemo/core/classes/modelPT.py", line 430, in save_to
    self._save_restore_connector.save_to(self, str(save_path))  # downstream tasks expect str, not Path
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/nemo/core/connectors/save_restore_connector.py", line 88, in save_to
    self._make_nemo_file_from_folder(filename=save_path, source_dir=tmpdir)
  File "/usr/local/lib/python3.11/site-packages/nemo/core/connectors/save_restore_connector.py", line 614, in _make_nemo_file_from_folder
    with tarfile.open(filename, "w:") as tar:
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/tarfile.py", line 1825, in open
    return func(name, filemode, fileobj, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/tarfile.py", line 1855, in taropen
    return cls(name, mode, fileobj, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/tarfile.py", line 1667, in __init__
    fileobj = bltn_open(name, self._mode)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/srv/models/finetuned_nemotron_v1_lr3e6_ep2.nemo'
