#!/bin/bash

mkdir -p .conda/cache
# Create conda env and re-direct package storage
CONDA_PKGS_DIRS=.conda/cache mamba create --prefix .conda/cipher --file env_simple.txt --yes python=3.10.12

# Activate env
mamba activate .conda/cipher
# Install packages
CONDA_PKGS_DIRS=.conda/cache mamba install scanpy pymc yaml --yes

# Add kernel to notebook
mamba activate .conda/cipher
python -m ipykernel install --user --name cipher
