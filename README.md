## A physics-informed cluster graph neural network (CG-NET)
This software package implements a physics-informed cluster graph neural network (CG-NET) designed to exploit short-range interactions and local structures through a cluster graph representation to facilitate material property prediction.

<img src="https://github.com/hchenglab/CG-NET/blob/main/assets/CGNET.png?raw=true" alt="CGNET" width="100%" style="float: right">

### Installation
CG-NET is built on the [Deep Graph Library (DGL)](https://www.dgl.ai), [Lightning](https://lightning.ai/docs/pytorch/stable/), and [PyTorch](https://pytorch.org), with specific adaptations designed for materials-based applications.

It is recommended to fetch the latest version of the main branch using:
```terminal
git clone https://github.com/hchenglab/CG-NET.git
```

Navigate to this folder, and the required packages can be installed into a conda environment using the [env.yml](./env.yml) file by running:
```terminal
conda env create -f env.yml
```

After the environment is created, install CG-NET via pip:
```terminal
conda activate cgnet
pip install .
```

#### CUDA (GPU) installation

If you intend to use CUDA (GPU) for faster training, it is important to install the appropriate versions of CUDA, PyTorch and DGL. For CUDA 12.1, the basic instructions are provided below, but it is recommended to consult the [PyTorch](https://pytorch.org) and [DGL](https://www.dgl.ai) documentation if you encounter any issues.
```terminal
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install  dgl -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html
```


