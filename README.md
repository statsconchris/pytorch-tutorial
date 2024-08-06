# PyTorch-tutorial

### Get the Driver version
`nvidia-smi` shows the CUDA version of your GPU, e.g.,
- GPU A: CUDA Version: 11.4
- GPU B: CUDA Version: 12.4

### Check compatible wheel files
Go to https://download.pytorch.org/whl/torch/

a)  GPU A

We see from the list that the highest compatible wheel file is `torch-1.12.1+cu113-cp310-cp310-linux_x86_64.whl`
 - cu113 = CUDA VERSION 11.3 < CUDA VERSION 11.4
 - cp310 = CPtyhon 3.10

b) GPU B

We see from the list that the highest compatible wheel file is `torch-2.4.0+cu124-cp312-cp312-linux_x86_64.whl`
 - cu124 = CUDA VERSION 12.4 <= CUDA VERSION 12.4
 - cp310 = CPtyhon 3.12

### Create and activate a conda environment with the correct Python version

a) GPU A

- `conda create -n <choose-your-name> python=3.10`
- `conda activate <choose-your-name>`

  
b) GPU B

- `conda create -n <choose-your-name> python=3.12` (if python 3.12 is the latest version then it is enough to type `python=3`)
- `conda activate <choose-your-name>`

*optional*
- `python --version` (to check python version)

### Install Pytorch from wheel

a) GPU A

- `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu113`

  **Note:** Python 3.10 or lower must be installed. Otherwise it will fail.

b) GPU B

- `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124`

*optional*
- `conda list` (to check installed packages), e.g., for GPU B we have:
  - torch                     2.4.0+cu124              
  - torchaudio                2.4.0+cu124              
  - torchvision               0.19.0+cu124 
 
### Set up the Jupyter kernel

In both cases:

- `conda install ipykernel -y && ipython kernel install --user --name "<choose-your-name>"`
  
*optional*
- `ls /home/<username>/.local/share/jupyter/kernels` (to list kernels)
- `jupyter kernelspec list` (to list kernels)

### Basic script to check that you are using a GPU

```
import torch

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# additional info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
```

### Remove the kernel and the conda environment

- `jupyter kernelspec uninstall <choose-your-name> -y`
- `conda remove -n <choose-your-name> --all -y`
- `conda clean --all`



