To activate CUDA (running tensorflow on GPU):
- Set up conda virtual environment with python version 3.7
- Run conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0

To install packages (and manage their versions):
- conda install tensorflow-gpu
- pip install pillow

To activate the tensorboard (outside of PyCharm):
- Anaconda Navigator install Powershell prompt
- Open Powershell prompt from environment in Anaconda Navigator
- cd to repo
- tensorboard --logdir logs