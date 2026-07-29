## A physically motivated cluster graph neural network (CG-NET)
This software package implements a physically motivated cluster graph neural network (CG-NET) designed to exploit short-range interactions and local structures through an enhanced cluster graph representation to facilitate material property prediction.

<img src="https://github.com/hchenglab/CG-NET/blob/main/assets/CGNET.png?raw=true" alt="CGNET" width="100%" style="float: right">

### Installation
CG-NET is backboned on the [fairchem](https://github.com/facebookresearch/fairchem) framework and [UMA](https://arxiv.org/abs/2506.23971) model.

It is recommended to fetch the latest version of the main branch using:
```bash
git clone https://github.com/hchenglab/CG-NET.git
cd CG-NET
```

The required packages can be installed into a virtual environment by running:
```bash
uv venv
source .venv/bin/activate
```

After the environment is created, install CG-NET via pip:
```bash
uv pip install -e packages/fairchem-core
```

### Datasets and Pretrained Models
The relevant datasets (e.g., HEA surface adsorption datasets) and pre-trained CG-NET model checkpoints used in our study are publicly available. 
You can download them from Figshare:
- **[Datasets & Pretrained Models](https://doi.org/10.6084/m9.figshare.29413064)**

Please download and place the dataset and checkpoint files into the appropriate directories before running the examples.

### Getting Started
CG-NET is an implementation of the cluster graph representation based on conventional crystal graph models, specifically the [fairchem](https://github.com/facebookresearch/fairchem) framework and [UMA](https://arxiv.org/abs/2506.23971) model. As such, the featurizer and convolutional operations in CG-NET follow the architecture of UMA. However, it is important to note that the cluster graph representation used in CG-NET is not confined to UMA. In fact, this approach can be extended to other advanced geometrically equivariant graph models or integrated with many-body interaction frameworks. Please refer to our paper for more details.

**Quick Run Example**
Below is a quick example of how to evaluate a model on the High Entropy Alloy (HEA) adsorption energy dataset (assuming you have set up the configs similar to our benchmark directory).

To run predictions using the local cluster graph (CG-NET) configuration:
```bash
uv run fairchem -c configs/cgnet/energy/test/local.yaml
```

To run predictions using the global graph configuration:
```bash
uv run fairchem -c configs/cgnet/energy/test/global.yaml
```
*(Note: Ensure your YAML configuration files properly reference the paths to the downloaded datasets and model checkpoints).*

### Citation
If you find this code, the pre-trained models, or our cluster graph formulation useful in your research, please cite our work:

```bibtex
@article{cgnet_under_review,
  title={A physically motivated cluster graph neural network for material property prediction (Title placeholder)},
  author={Author names placeholder},
  journal={Under Review},
  year={2026}
}
```

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.