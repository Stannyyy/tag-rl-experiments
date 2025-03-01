To activate the tensorboard (outside of PyCharm):
- Anaconda Navigator install Powershell prompt
- Open Powershell prompt from environment in Anaconda Navigator
- cd to repo
- tensorboard --logdir logs

To activate CUDA (running tensorflow on GPU):
- Download and install cuda toolkit
- Run
    pip install nvidia-pyindex
    pip install nvidia-cudnn