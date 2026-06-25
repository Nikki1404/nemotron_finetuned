
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
28.9 M    Trainable params
609 M     Non-trainable params
637 M     Total params
2,551.988 Total estimated model params size (MB)
710       Modules in train mode
0         Modules in eval mode
[NeMo W 2026-06-24 22:00:44 nemo_logging:364] /usr/local/lib/python3.11/site-packages/lightning/pytorch/trainer/connectors/data_connector.py:424: The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=7` in the `DataLoader` to improve performance.

[NeMo W 2026-06-24 22:00:44 nemo_logging:364] /usr/local/lib/python3.11/site-packages/lightning/pytorch/trainer/connectors/data_connector.py:424: The 'val_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=7` in the `DataLoader` to improve performance.

Epoch 0:   0%|                                                                                                | 0/5 [00:00<?, ?it/s][NeMo I 2026-06-24 22:00:44 asr_model:198] CUDA graphs disabled for EncDecRNNTBPEModelWithPrompt::RNNTBPEDecoding::GreedyBatchedRNNTInfer
Traceback (most recent call last):
  File "/workspace/scripts/finetune_nemotron.py", line 213, in <module>
    main()
  File "/workspace/scripts/finetune_nemotron.py", line 202, in main
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
  File "/usr/local/lib/python3.11/site-packages/lightning/pytorch/trainer/trainer.py", line 1025, in _run_stage
    self.fit_loop.run()
  File "/usr/local/lib/python3.11/site-packages/lightning/pytorch/loops/fit_loop.py", line 205, in run
    self.advance()
  File "/usr/local/lib/python3.11/site-packages/lightning/pytorch/loops/fit_loop.py", line 363, in advance
    self.epoch_loop.run(self._data_fetcher)
  File "/usr/local/lib/python3.11/site-packages/lightning/pytorch/loops/training_epoch_loop.py", line 140, in run
    self.advance(data_fetcher)
  File "/usr/local/lib/python3.11/site-packages/lightning/pytorch/loops/training_epoch_loop.py", line 250, in advance
    batch_output = self.automatic_optimization.run(trainer.optimizers[0], batch_idx, kwargs)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/lightning/pytorch/loops/optimization/automatic.py", line 190, in run
    self._optimizer_step(batch_idx, closure)
  File "/usr/local/lib/python3.11/site-packages/lightning/pytorch/loops/optimization/automatic.py", line 268, in _optimizer_step
    call._call_lightning_module_hook(
  File "/usr/local/lib/python3.11/site-packages/lightning/pytorch/trainer/call.py", line 167, in _call_lightning_module_hook
    output = fn(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/lightning/pytorch/core/module.py", line 1306, in optimizer_step
    optimizer.step(closure=optimizer_closure)
  File "/usr/local/lib/python3.11/site-packages/lightning/pytorch/core/optimizer.py", line 153, in step
    step_output = self._strategy.optimizer_step(self._optimizer, closure, **kwargs)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/lightning/pytorch/strategies/strategy.py", line 238, in optimizer_step
    return self.precision_plugin.optimizer_step(optimizer, model=model, closure=closure, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/lightning/pytorch/plugins/precision/amp.py", line 75, in optimizer_step
    return super().optimizer_step(optimizer, model=model, closure=closure, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/lightning/pytorch/plugins/precision/precision.py", line 122, in optimizer_step
    return optimizer.step(closure=closure, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/torch/optim/lr_scheduler.py", line 140, in wrapper
    return func.__get__(opt, opt.__class__)(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/torch/optim/optimizer.py", line 493, in wrapper
    out = func(*args, **kwargs)
          ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/torch/optim/optimizer.py", line 91, in _use_grad
    ret = func(self, *args, **kwargs)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/torch/optim/adamw.py", line 220, in step
    loss = closure()
           ^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/lightning/pytorch/plugins/precision/precision.py", line 108, in _wrap_closure
    closure_result = closure()
                     ^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/lightning/pytorch/loops/optimization/automatic.py", line 144, in __call__
    self._result = self.closure(*args, **kwargs)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/torch/utils/_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/lightning/pytorch/loops/optimization/automatic.py", line 129, in closure
    step_output = self._step_fn()
                  ^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/lightning/pytorch/loops/optimization/automatic.py", line 317, in _training_step
    training_step_output = call._call_strategy_hook(trainer, "training_step", *kwargs.values())
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/lightning/pytorch/trainer/call.py", line 319, in _call_strategy_hook
    output = fn(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/lightning/pytorch/strategies/strategy.py", line 390, in training_step
    return self.lightning_module.training_step(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/scripts/finetune_nemotron.py", line 99, in new_training_step
    return old_training_step(batch, batch_idx)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/nemo/utils/model_utils.py", line 471, in wrap_training_step
    output_dict = wrapped(*args, **kwargs)
                  ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/nemo/collections/asr/models/rnnt_bpe_models_prompt.py", line 418, in training_step
    loss_value, wer, _, _ = self.joint(
                            ^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/nemo/core/classes/common.py", line 1360, in wrapped_call
    outputs = wrapped(*args, **kwargs)
              ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/nemo/collections/asr/modules/rnnt.py", line 1560, in forward
    sub_joint = self.joint(sub_enc, sub_dec)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/nemo/collections/asr/modules/rnnt_abstract.py", line 100, in joint
    return self.joint_after_projection(self.project_encoder(f), self.project_prednet(g))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/nemo/collections/asr/modules/rnnt.py", line 1724, in joint_after_projection
    res = self.joint_net(inp)  # [B, T, U, V + 1]
          ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/torch/nn/modules/container.py", line 250, in forward
    input = module(input)
            ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/torch/nn/modules/linear.py", line 125, in forward
    return F.linear(input, self.weight, self.bias)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 18.51 GiB. GPU 0 has a total capacity of 22.09 GiB of which 7.65 GiB is free. Including non-PyTorch memory, this process has 0 bytes memory in use. Of the allocated memory 4.80 GiB is allocated by PyTorch, and 22.38 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
Epoch 0:   0%|          | 0/5 [00:02<?, ?it/s]
root@c6f79d6e94db:/workspace# client_loop: send disconnect: Connection reset
