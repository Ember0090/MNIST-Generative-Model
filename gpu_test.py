# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 13:29:50 2025

@author: joshua
"""


# %% environment setup

# All done in WSL

# INSTALL CUDA SUPPORT
# https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu

# CREATE THE ENVIRONMENT
# conda create -n aigpu
# conda activate aigpu

# INSTALL PACKAGES
# conda install pip
# pip install numpy torch keras matplotlib spyder-kernels ipykernel
# pip install "tf-nightly[and-cuda]"
    # needed for 50-series GPUs for the time being, otherwise, 
    # pip install tensorflow[and-cuda]


# %% wsl gpu

# start the spyder kernel in wsl:
# conda activate tfgpu
# python -m spyder_kernels.console

# access it using the Consoles > existing kernel:
# //wsl$/Ubuntu/home/joshua/.local/share/jupyter/runtime/kernel-****.json


# %% check wsl kernal works

import sys, platform, os
print(sys.executable)      # should point to /home/.../miniconda3/envs/myenv/bin/python
print(platform.platform()) # should say Linux
os.system("uname -a")      # should return Linux kernel info


# %% pytorch gpu test

import torch
print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("GPU available?", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(0))
    # simple test tensor
    x = torch.rand(3, 3).to("cuda")
    print("Tensor on GPU:", x)


# %% example of pytorch model run on gpu

import torch

# Pick device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example: create a tensor on GPU
x = torch.rand(3, 3, device=device)

# Move an existing tensor/model
x_cpu = torch.rand(3, 3)
x_gpu = x_cpu.to(device)

# Example: model
model = MyModel().to(device)
output = model(x_gpu)


# %% tensorflow gpu test

import tensorflow as tf
print("TF version:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))
# Run a simple matmul on GPU
with tf.device("/GPU:0"):
    a = tf.random.normal([1000, 1000])
    b = tf.random.normal([1000, 1000])
    c = tf.matmul(a, b)
print("Matrix multiplication result shape:", c.shape)


# %% example of tensorflow model run on gpu

import tensorflow as tf

# List devices
print(tf.config.list_physical_devices('GPU'))

# Explicit GPU usage
with tf.device('/GPU:0'):
    a = tf.random.normal([1000, 1000])
    b = tf.random.normal([1000, 1000])
    c = tf.matmul(a, b)
print("OK:", c.shape)

# %%


device = "cuda" if torch.cuda.is_available() else "cpu"
tf.config.list_physical_devices('GPU')

# Allow memory growth (prevents TF from grabbing all GPU RAM)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


