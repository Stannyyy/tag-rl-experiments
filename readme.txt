To activate CUDA (running tensorflow on GPU):
- Set up conda virtual environment with python version 3.7
- Follow instructions on: https://www.tensorflow.org/install/pip#windows-native
    conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
    # Anything above 2.10 is not supported on the GPU on Windows Native
    python -m pip install "tensorflow<2.11"
    # Verify the installation:
    python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

To install packages (and manage their versions):
- pip install pillow

To activate the tensorboard (outside of PyCharm):
- Anaconda Navigator install Powershell prompt
- Open Powershell prompt from environment in Anaconda Navigator
- cd to repo
- tensorboard --logdir experiment-<date>-<time>/logs