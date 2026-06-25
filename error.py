[train] Starting fine-tuning...
You are using a CUDA device ('NVIDIA A10G') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
[NeMo I 2026-06-25 10:32:14 modelPT:781] Optimizer config = AdamW (
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
[NeMo I 2026-06-25 10:32:14 lr_scheduler:995] Scheduler "<nemo.core.optim.lr_scheduler.CosineAnnealing object at 0x7f6f190ddc50>"
    will be used during training (effective maximum steps = 105) -
    Parameters :
    (warmup_steps: 5
    min_lr: 2.5000000000000004e-07
    max_steps: 105
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
28.9 M    Trainable params
609 M     Non-trainable params
637 M     Total params
2,551.988 Total estimated model params size (MB)
710       Modules in train mode
0         Modules in eval mode
[NeMo W 2026-06-25 10:32:16 nemo_logging:364] /usr/local/lib/python3.11/site-packages/lightning/pytorch/trainer/connectors/data_connector.py:424: The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=7` in the `DataLoader` to improve performance.

[NeMo W 2026-06-25 10:32:16 nemo_logging:364] /usr/local/lib/python3.11/site-packages/lightning/pytorch/trainer/connectors/data_connector.py:424: The 'val_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=7` in the `DataLoader` to improve performance.

Epoch 0:   0%|                                                                                               | 0/35 [00:00<?, ?it/s][NeMo I 2026-06-25 10:32:16 asr_model:198] CUDA graphs disabled for EncDecRNNTBPEModelWithPrompt::RNNTBPEDecoding::GreedyBatchedRNNTInfer
[NeMo W 2026-06-25 10:32:17 rnnt:456] Provided RNNT Joint tensor is of dtype torch.float16, but RNNT loss could not be calculated in fp16 due to following reason stated below. Loss will be calculated in fp32.

    Env variable `NUMBA_CUDA_USE_NVIDIA_BINDING` is not available or has not set to `1`.Numba CUDA FP16 is supported in installed numba version.
Traceback (most recent call last):
  File "/usr/local/lib/python3.11/site-packages/numba/cuda/cudadrv/nvvm.py", line 139, in __new__
    inst.driver = open_cudalib('nvvm')
                  ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/cuda/cudadrv/libs.py", line 65, in open_cudalib
    return ctypes.CDLL(path)
           ^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/ctypes/__init__.py", line 376, in __init__
    self._handle = _dlopen(self._name, mode)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^
OSError: libnvvm.so: cannot open shared object file: No such file or directory

During handling of the above exception, another exception occurred:

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
  File "/usr/local/lib/python3.11/site-packages/lightning/pytorch/plugins/precision/amp.py", line 78, in optimizer_step
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
  File "/usr/local/lib/python3.11/site-packages/nemo/collections/asr/modules/rnnt.py", line 1577, in forward
    loss_batch = self.loss(
                 ^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/nemo/core/classes/common.py", line 1360, in wrapped_call
    outputs = wrapped(*args, **kwargs)
              ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/nemo/collections/asr/losses/rnnt.py", line 488, in forward
    loss = self._loss(acts=log_probs, labels=targets, act_lens=input_lengths, label_lens=target_lengths)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/nemo/collections/asr/parts/numba/rnnt_loss/rnnt_pytorch.py", line 437, in forward
    return self.loss(
           ^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/torch/autograd/function.py", line 575, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/nemo/collections/asr/parts/numba/rnnt_loss/rnnt_pytorch.py", line 62, in forward
    loss_func(
  File "/usr/local/lib/python3.11/site-packages/nemo/collections/asr/parts/numba/rnnt_loss/rnnt.py", line 223, in rnnt_loss_gpu
    status = wrapper.cost_and_grad(
             ^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/nemo/collections/asr/parts/numba/rnnt_loss/utils/cuda_utils/gpu_rnnt.py", line 252, in cost_and_grad
    return self.compute_cost_and_score(acts, grads, costs, pad_labels, label_lengths, input_lengths)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/nemo/collections/asr/parts/numba/rnnt_loss/utils/cuda_utils/gpu_rnnt.py", line 161, in compute_cost_and_score
    self.log_softmax(acts, denom)
  File "/usr/local/lib/python3.11/site-packages/nemo/collections/asr/parts/numba/rnnt_loss/utils/cuda_utils/gpu_rnnt.py", line 107, in log_softmax
    reduce.reduce_max(
  File "/usr/local/lib/python3.11/site-packages/nemo/collections/asr/parts/numba/rnnt_loss/utils/cuda_utils/reduce.py", line 353, in reduce_max
    return ReduceHelper(
           ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/nemo/collections/asr/parts/numba/rnnt_loss/utils/cuda_utils/reduce.py", line 294, in ReduceHelper
    _reduce_rows[grid_size, CTA_REDUCE_SIZE, stream, 0](I_opid, R_opid, acts, output, num_rows)
  File "/usr/local/lib/python3.11/site-packages/numba/cuda/dispatcher.py", line 539, in __call__
    return self.dispatcher.call(args, self.griddim, self.blockdim,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/cuda/dispatcher.py", line 681, in call
    kernel = _dispatcher.Dispatcher._cuda_call(self, *args)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/cuda/dispatcher.py", line 689, in _compile_for_args
    return self.compile(tuple(argtypes))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/cuda/dispatcher.py", line 932, in compile
    kernel = _Kernel(self.py_func, argtypes, **self.targetoptions)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/core/compiler_lock.py", line 35, in _acquire_compile_lock
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/cuda/dispatcher.py", line 83, in __init__
    cres = compile_cuda(self.py_func, types.void, self.argtypes,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/core/compiler_lock.py", line 35, in _acquire_compile_lock
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/cuda/compiler.py", line 196, in compile_cuda
    cres = compiler.compile_extra(typingctx=typingctx,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/core/compiler.py", line 739, in compile_extra
    return pipeline.compile_extra(func)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/core/compiler.py", line 439, in compile_extra
    return self._compile_bytecode()
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/core/compiler.py", line 505, in _compile_bytecode
    return self._compile_core()
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/core/compiler.py", line 481, in _compile_core
    raise e
  File "/usr/local/lib/python3.11/site-packages/numba/core/compiler.py", line 473, in _compile_core
    pm.run(self.state)
  File "/usr/local/lib/python3.11/site-packages/numba/core/compiler_machinery.py", line 363, in run
    raise e
  File "/usr/local/lib/python3.11/site-packages/numba/core/compiler_machinery.py", line 356, in run
    self._runPass(idx, pass_inst, state)
  File "/usr/local/lib/python3.11/site-packages/numba/core/compiler_lock.py", line 35, in _acquire_compile_lock
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/core/compiler_machinery.py", line 311, in _runPass
    mutated |= check(pss.run_pass, internal_state)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/core/compiler_machinery.py", line 272, in check
    mangled = func(compiler_state)
              ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/core/typed_passes.py", line 114, in run_pass
    typemap, return_type, calltypes, errs = type_inference_stage(
                                            ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/core/typed_passes.py", line 95, in type_inference_stage
    errs = infer.propagate(raise_errors=raise_errors)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/core/typeinfer.py", line 1075, in propagate
    errors = self.constraints.propagate(self)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/core/typeinfer.py", line 160, in propagate
    constraint(typeinfer)
  File "/usr/local/lib/python3.11/site-packages/numba/core/typeinfer.py", line 572, in __call__
    self.resolve(typeinfer, typevars, fnty)
  File "/usr/local/lib/python3.11/site-packages/numba/core/typeinfer.py", line 595, in resolve
    sig = typeinfer.resolve_call(fnty, pos_args, kw_args)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/core/typeinfer.py", line 1569, in resolve_call
    return self.context.resolve_function_type(fnty, pos_args, kw_args)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/core/typing/context.py", line 279, in resolve_function_type
    res = self._resolve_user_function_type(func, args, kws)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/core/typing/context.py", line 335, in _resolve_user_function_type
    return func.get_call_type(self, args, kws)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/core/types/functions.py", line 541, in get_call_type
    self.dispatcher.get_call_template(args, kws)
  File "/usr/local/lib/python3.11/site-packages/numba/cuda/dispatcher.py", line 849, in get_call_template
    self.compile_device(tuple(args))
  File "/usr/local/lib/python3.11/site-packages/numba/cuda/dispatcher.py", line 883, in compile_device
    cres = compile_cuda(self.py_func, return_type, args,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/core/compiler_lock.py", line 35, in _acquire_compile_lock
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/cuda/compiler.py", line 196, in compile_cuda
    cres = compiler.compile_extra(typingctx=typingctx,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/core/compiler.py", line 739, in compile_extra
    return pipeline.compile_extra(func)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/core/compiler.py", line 439, in compile_extra
    return self._compile_bytecode()
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/core/compiler.py", line 505, in _compile_bytecode
    return self._compile_core()
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/core/compiler.py", line 481, in _compile_core
    raise e
  File "/usr/local/lib/python3.11/site-packages/numba/core/compiler.py", line 473, in _compile_core
    pm.run(self.state)
  File "/usr/local/lib/python3.11/site-packages/numba/core/compiler_machinery.py", line 363, in run
    raise e
  File "/usr/local/lib/python3.11/site-packages/numba/core/compiler_machinery.py", line 356, in run
    self._runPass(idx, pass_inst, state)
  File "/usr/local/lib/python3.11/site-packages/numba/core/compiler_lock.py", line 35, in _acquire_compile_lock
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/core/compiler_machinery.py", line 311, in _runPass
    mutated |= check(pss.run_pass, internal_state)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/core/compiler_machinery.py", line 272, in check
    mangled = func(compiler_state)
              ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/core/typed_passes.py", line 470, in run_pass
    lower = self.lowering_class(targetctx, library, fndesc, interp,
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/core/lowering.py", line 40, in __init__
    self.module = self.library.create_ir_module(self.fndesc.unique_name)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/core/codegen.py", line 576, in create_ir_module
    ir_module = self._codegen._create_empty_module(name)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/cuda/codegen.py", line 365, in _create_empty_module
    ir_module.data_layout = nvvm.NVVM().data_layout
                            ^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/cuda/cudadrv/nvvm.py", line 144, in __new__
    raise NvvmSupportError(errmsg % e)
numba.cuda.cudadrv.error.NvvmSupportError: libNVVM cannot be found. Do `conda install cudatoolkit`:
libnvvm.so: cannot open shared object file: No such file or directory
Epoch 0:   0%|          | 0/35 [00:02<?, ?it/s]
