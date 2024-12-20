## A physics-informed cluster graph neural network (CG-NET)
This software package implements a physics-informed cluster graph neural network (CG-NET) designed to exploit short-range interactions and local structures through a cluster graph representation to facilitate material property prediction.

<img src="https://github.com/hchenglab/CG-NET/blob/main/assets/CGNET.png?raw=true" alt="CGNET" width="100%" style="float: right">

### Installation
CG-NET is built on the [Deep Graph Library (DGL)](https://www.dgl.ai), [Lightning](https://lightning.ai/docs/pytorch/stable/), and [PyTorch](https://pytorch.org), with specific adaptations designed for materials-based applications.

It is recommended to fetch the latest version of the main branch using:
```terminal
git clone https://github.com/hchenglab/CG-NET.git
```

Navigate to this folder, and the required packages can be installed into a conda environment using the `env.yml` file by running:
```terminal
conda env create -f env.yml
```

After the environment is created, install CG-NET via pip:
```terminal
conda activate cgnet
pip install .
```

#### CUDA (GPU) Installation

If you intend to use CUDA (GPU) for faster training, it is important to install the appropriate versions of CUDA, PyTorch and DGL. For CUDA 12.1, the basic instructions are provided below, but it is recommended to consult the [PyTorch](https://pytorch.org) and [DGL](https://www.dgl.ai) documentation if you encounter any issues.
```terminal
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install  dgl -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html
```

### Getting Started
CG-NET is an implementation of the cluster graph representation based on conventional crystal graph models, specifically [CGCNN (Crystal Graph Convolutional Neural Networks)](https://github.com/txie-93/cgcnn/tree/master). As such, the featurizer and convolutional operations in CG-NET follow the architecture of CGCNN. However, it is important to note that the cluster graph representation used in CG-NET is not confined to CGCNN, which typically focuses on pair-wise interactions through an invariant graph model. In fact, this approach can be extended to more advanced geometrically equivariant graph models or integrated with many-body interaction frameworks.

#### Define a customized dataset
To defining a customized dataset, you will need:
- [CIF](https://en.wikipedia.org/wiki/Crystallographic_Information_File) files containing the crystal structure data of the materials you are interested in
- The target properties associated with each crystal.
- The atom indexes of the cluster centers you want to define in each crystal

You can create a customized dataset by organizing your data into a directory called `raw_dataset`, which should contain the following files:

1. `id_prop_index.csv`: a [CSV](https://en.wikipedia.org/wiki/Comma-separated_values) file with at least three columns. 
    - The first column should contain a unique `ID` for each crystal 
    - The second column should contain the value of the target property for each crystal
    - Starting from the third column, the file should list the atom indexes of the cluster centers for each crystal. If you are studying systems with multiple cluster centers, you can define more than one cluster center by adding additional columns for each cluster center’s atom index

2. `atom_init.json`: a [JSON](https://en.wikipedia.org/wiki/JSON) file that stores the initialization vector for each element. 
The `atom_init.json`, inherited from [CGCNN](https://github.com/txie-93/cgcnn/tree/master), should work for most applications.

3. `ID.cif`: a [CIF](https://en.wikipedia.org/wiki/Crystallographic_Information_File) file that recodes the crystal structure, where `ID` is the unique `ID` for the crystal.

The structure of the `raw_dataset` should be:

```
raw_dataset
├── id_prop_index.csv
├── atom_init.json
├── id0.cif
├── id1.cif
├── ...
```

#### Train a CG-NET model

Then, you can train a CG-NET model for your customized dataset by:

```terminal
python train.py 
```
In `train.py`, you will also find a few adjustable parameters regarding the CG-NET architecture and training.

After training, you will get three files in `logs/cgnet/version_x` directory.

- `checkpoints`: stores the checkpoints of the CG-NET model with the best validation loss.
- `hparams.yaml`: stores the hyperparameters used for training.
- `test_results.csv`: stores the `ID`, target value, and predicted value for each crystal in test set.

#### Predict material properties with a pre-trained CGCNN model
After training, you can use the pre-trained model to predict the properties of new materials. To do this, you will need to run `predict.py`.
```terminal
python predict.py
```
After predicting, you will get `predict_results.csv` in the current directory.

- `predict_results.csv`: stores the `ID` and predicted value for each crystal in the prediction set.

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.