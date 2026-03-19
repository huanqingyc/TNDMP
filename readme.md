# TNDMP

This repository contains the implementation of the algorithm described in the paper *Tensor Network Dynamical Message Passing for epidemic models*, together with the code needed to reproduce the data, figures, and experiments reported there.

It is intended for research reproduction and reference use, rather than as a polished general-purpose software package.

## Environment Setup

Create the environment with Conda:

```bash
conda env create -f environment.yml
conda activate tndmp
```

Install PyTorch separately for your local platform and hardware configuration after creating the environment.

## Reproducing Results

Run the reproduction script to regenerate the paper outputs:

```bash
bash reproduce.sh
```

Runtime, numerical behavior, and produced outputs may vary slightly depending on the local environment and the specific PyTorch build in use.