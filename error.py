root@c6f79d6e94db:/workspace# apt update
Hit:1 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64  InRelease
Hit:2 http://archive.ubuntu.com/ubuntu jammy InRelease
Get:3 http://security.ubuntu.com/ubuntu jammy-security InRelease [129 kB]
Get:4 http://archive.ubuntu.com/ubuntu jammy-updates InRelease [128 kB]
Hit:5 http://archive.ubuntu.com/ubuntu jammy-backports InRelease
Fetched 257 kB in 1s (240 kB/s)
Reading package lists... Done
Building dependency tree... Done
Reading state information... Done
2 packages can be upgraded. Run 'apt list --upgradable' to see them.
root@c6f79d6e94db:/workspace# apt install -y cuda-nvvm-12-4
Reading package lists... Done
Building dependency tree... Done
Reading state information... Done
The following NEW packages will be installed:
  cuda-nvvm-12-4
0 upgraded, 1 newly installed, 0 to remove and 2 not upgraded.
Need to get 19.5 MB of archives.
After this operation, 64.5 MB of additional disk space will be used.
Get:1 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64  cuda-nvvm-12-4 12.4.131-1 [19.5 MB]
Fetched 19.5 MB in 1s (37.0 MB/s)
debconf: delaying package configuration, since apt-utils is not installed
Selecting previously unselected package cuda-nvvm-12-4.
(Reading database ... 14653 files and directories currently installed.)
Preparing to unpack .../cuda-nvvm-12-4_12.4.131-1_amd64.deb ...
Unpacking cuda-nvvm-12-4 (12.4.131-1) ...
Setting up cuda-nvvm-12-4 (12.4.131-1) ...
root@c6f79d6e94db:/workspace# find /usr/local/cuda* -name libnvvm.so
/usr/local/cuda-12.4/nvvm/lib64/libnvvm.so
root@c6f79d6e94db:/workspace# export LD_LIBRARY_PATH=/usr/local/cuda-12.4/nvvm/lib64:$LD_LIBRARY_PATH
root@c6f79d6e94db:/workspace# export NUMBA_CUDA_USE_NVIDIA_BINDING=1
root@c6f79d6e94db:/workspace# NUMBA_CUDA_USE_NVIDIA_BINDING=1 \
LD_LIBRARY_PATH=/usr/local/cuda-12.4/nvvm/lib64:$LD_LIBRARY_PATH \
python3.11 scripts/finetune_nemotron.py \
  --train-manifest data/manifests/train_manifest.json \
  --val-manifest data/manifests/val_manifest.json \
  --base-model /srv/nemotron-3.5-asr-streaming-0.6b.nemo \
  --output-nemo /srv/models/nemotron_inspira_decoder_ft.nemo \
  --freeze-mode decoder_only \
  --max-epochs 5 \
  --batch-size 1 \
  --lr 5e-6 \
  --language en-US \
  --precision bf16-mixed
OneLogger: Setting error_handling_strategy to DISABLE_QUIETLY_AND_REPORT_METRIC_ERROR for rank (rank=0) with OneLogger disabled. To override: explicitly set error_handling_strategy parameter.
No exporters were provided. This means that no telemetry data will be collected.
Traceback (most recent call last):
  File "/workspace/scripts/finetune_nemotron.py", line 213, in <module>
    main()
  File "/workspace/scripts/finetune_nemotron.py", line 137, in main
    model = load_model(args.base_model, args.language)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/scripts/finetune_nemotron.py", line 12, in load_model
    import nemo.collections.asr as nemo_asr
  File "/usr/local/lib/python3.11/site-packages/nemo/collections/asr/__init__.py", line 15, in <module>
    from nemo.collections.asr import data, losses, models, modules
  File "/usr/local/lib/python3.11/site-packages/nemo/collections/asr/models/__init__.py", line 15, in <module>
    from nemo.collections.asr.models.aed_multitask_models import EncDecMultiTaskModel  # noqa: F401
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/nemo/collections/asr/models/aed_multitask_models.py", line 32, in <module>
    from nemo.collections.asr.metrics import MultiTaskMetric
  File "/usr/local/lib/python3.11/site-packages/nemo/collections/asr/metrics/__init__.py", line 15, in <module>
    from nemo.collections.asr.metrics.bleu import BLEU
  File "/usr/local/lib/python3.11/site-packages/nemo/collections/asr/metrics/bleu.py", line 23, in <module>
    from nemo.collections.asr.parts.submodules.ctc_decoding import AbstractCTCDecoding
  File "/usr/local/lib/python3.11/site-packages/nemo/collections/asr/parts/submodules/ctc_decoding.py", line 25, in <module>
    from nemo.collections.asr.parts.submodules import ctc_beam_decoding, ctc_greedy_decoding
  File "/usr/local/lib/python3.11/site-packages/nemo/collections/asr/parts/submodules/ctc_beam_decoding.py", line 24, in <module>
    from nemo.collections.asr.parts.context_biasing import BoostingTreeModelConfig, GPUBoostingTreeModel
  File "/usr/local/lib/python3.11/site-packages/nemo/collections/asr/parts/context_biasing/__init__.py", line 15, in <module>
    from nemo.collections.asr.parts.context_biasing.boosting_graph_batched import (
  File "/usr/local/lib/python3.11/site-packages/nemo/collections/asr/parts/context_biasing/boosting_graph_batched.py", line 28, in <module>
    from nemo.collections.asr.parts.submodules.ngram_lm import DEFAULT_TOKEN_OFFSET, NGramGPULanguageModel
  File "/usr/local/lib/python3.11/site-packages/nemo/collections/asr/parts/submodules/ngram_lm/__init__.py", line 17, in <module>
    from nemo.collections.asr.parts.submodules.ngram_lm.ngram_lm_batched import KenLMBatchedWrapper, NGramGPULanguageModel
  File "/usr/local/lib/python3.11/site-packages/nemo/collections/asr/parts/submodules/ngram_lm/ngram_lm_batched.py", line 33, in <module>
    from nemo.core.utils.optional_libs import KENLM_AVAILABLE, TRITON_AVAILABLE, kenlm_required, triton_required
  File "/usr/local/lib/python3.11/site-packages/nemo/core/utils/optional_libs.py", line 71, in <module>
    NUMBA_CUDA_AVAILABLE = numba_cuda_is_supported(__NUMBA_MINIMUM_VERSION__)
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/nemo/core/utils/numba_utils.py", line 129, in numba_cuda_is_supported
    from numba import cuda
  File "/usr/local/lib/python3.11/site-packages/numba/cuda/__init__.py", line 7, in <module>
    from .device_init import *
  File "/usr/local/lib/python3.11/site-packages/numba/cuda/device_init.py", line 3, in <module>
    from numba.cuda import cg
  File "/usr/local/lib/python3.11/site-packages/numba/cuda/cg.py", line 4, in <module>
    from numba.cuda import nvvmutils
  File "/usr/local/lib/python3.11/site-packages/numba/cuda/nvvmutils.py", line 4, in <module>
    from .cudadrv import nvvm
  File "/usr/local/lib/python3.11/site-packages/numba/cuda/cudadrv/nvvm.py", line 16, in <module>
    from .libs import get_libdevice, open_libdevice, open_cudalib
  File "/usr/local/lib/python3.11/site-packages/numba/cuda/cudadrv/libs.py", line 19, in <module>
    from numba.cuda.cudadrv.driver import locate_driver_and_loader, load_driver
  File "/usr/local/lib/python3.11/site-packages/numba/cuda/cudadrv/driver.py", line 43, in <module>
    from cuda import cuda as binding
ImportError: cannot import name 'cuda' from 'cuda' (unknown location)
