# GPU Training

## Environment Setup

* PyTorch 1.12 with Cuda 10.0.13

### Nvidia GPU and Cuda

Cuda 10.0.13
https://developer.nvidia.com/cuda-10.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1604&target_type=runfilelocal

nvidia command

$ nvcc --version
$ nvidia-smi

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
.to(device)
https://github.com/pytorch/pytorch/issues/1668

CUDA_VISIBLE_DEVICES

[CUDA Pro Tip: Control GPU Visibility with CUDA_VISIBLE_DEVICES | NVIDIA Developer Blog](https://devblogs.nvidia.com/cuda-pro-tip-control-gpu-visibility-cuda_visible_devices/)
[cuda - How do I select which GPU to run a job on? - Stack Overflow](https://stackoverflow.com/questions/39649102/how-do-i-select-which-gpu-to-run-a-job-on)
[Set Default GPU in PyTorch - jdhao's blog](https://jdhao.github.io/2018/04/02/pytorch-gpu-usage/)
[torch.cuda — PyTorch master documentation](https://pytorch.org/docs/stable/cuda.html)
[Syllo/nvtop: NVIDIA GPUs htop like monitoring tool](https://github.com/Syllo/nvtop)

### Pytorch version

PyTorch 1.12

## Pytorch

### Single GPU

### Multiple GPU

[Multi-GPU Examples — PyTorch Tutorials 1.2.0 documentation](https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html)
[Optional: Data Parallelism — PyTorch Tutorials 1.2.0 documentation](https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html)
[torch.nn — PyTorch master documentation](https://pytorch.org/docs/stable/nn.html#dataparallel-layers-multi-gpu-distributed)

[Run Pytorch on Multiple GPUs - PyTorch Forums](https://discuss.pytorch.org/t/run-pytorch-on-multiple-gpus/20932/17)


RNN warning

[RNN module weights are not part of single contiguous chunk of memory - PyTorch Forums](https://discuss.pytorch.org/t/rnn-module-weights-are-not-part-of-single-contiguous-chunk-of-memory/6011)

```py
# add
self.rnn.flatten_parameters()
# before
output, hn = self.rnn(x)
```
