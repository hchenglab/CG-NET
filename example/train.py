import os
import gc
import csv
import time
import argparse
import warnings
import random

from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor

import torch
from dgl.data.utils import split_dataset
from dgl.dataloading import GraphDataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from cgnet.data import Featureizer, CGCNNDataset
from cgnet.model import CGCNN

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(
    description="Cluster Graph Convolutional Neural Networks"
)

parser.add_argument(
    "--exp-name",
    default="cgnet",
    type=str,
    metavar="cgnet",
    help="name of the experiment",
)
parser.add_argument(
    "-j",
    default=0,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 0)",
)
parser.add_argument(
    "--epochs",
    default=500,
    type=int,
    metavar="N",
    help="number of total epochs to run (default: 30)",
)
parser.add_argument(
    "-b",
    default=64,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256)",
)
parser.add_argument(
    "--task",
    choices=["regression", "classification"],
    default="regression",
    type=str,
    metavar="T",
    help="complete a regression or " "classification task (default: regression)",
)
parser.add_argument(
    "--lr",
    default=0.01,
    type=float,
    metavar="LR",
    help="initial learning rate (default: " "0.01)",
)
parser.add_argument(
    "--hidden-node-dim",
    default=64,
    type=int,
    metavar="N",
    help="number of hidden atom features in conv layers",
)
parser.add_argument(
    "--predictor-hidden-feats",
    default=128,
    type=int,
    metavar="N",
    help="number of hidden features after pooling",
)
parser.add_argument(
    "--num-conv", default=3, type=int, metavar="N", help="number of conv layers"
)
parser.add_argument(
    "--max-neighbors",
    default=12,
    type=int,
    metavar="N",
    help="maximum number of neighbors for each node" "considered (default: 12)",
)
parser.add_argument(
    "--radius",
    default=8.0,
    type=float,
    metavar="N",
    help="radius for searching neighbors (default: 8.0)",
)
parser.add_argument(
    "--cluster-radius",
    default=10.0,
    type=float,
    metavar="N",
    help="radius for searching cluster atoms (default: 10.0)",
)
parser.add_argument(
    "--max-nodes",
    default=12,
    type=int,
    metavar="N",
    help="maximum number of nodes in the cluster graph",
)
parser.add_argument(
    "--train-ratio",
    default=0.8,
    type=float,
    metavar="N",
    help="number of training data to be loaded (default 0.8)",
)
parser.add_argument(
    "--val-ratio",
    default=0.1,
    type=float,
    metavar="N",
    help="percentage of validation data to be loaded (default 0.1)",
)
parser.add_argument(
    "--test-ratio",
    default=0.1,
    type=float,
    metavar="N",
    help="percentage of test data to be loaded (default 0.1)",
)

args = parser.parse_args()

seed = 42
pl.seed_everything(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"

start_time = time.time()

cif_path = "raw_dataset"


def gendata(path: str):
    adaptor = AseAtomsAdaptor()
    # read csv file
    ids, cidxs, energies = [], [], []
    with open(os.path.join(path, "id_prop_index.csv"), "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            ids.append(row[0])
            energies.append(float(row[1]))
            cidxs.append([int(i) for i in row[2:]])

    structures = [
        adaptor.get_structure(read(os.path.join(path, f"{id}.cif"))) for id in ids
    ]
    return ids, cidxs, structures, energies


# prepare the dataset
ids, cidxs, structures, energies = gendata(cif_path)

# set featureizer
featurizer = Featureizer(
    args.radius, args.max_neighbors, args.cluster_radius, args.max_nodes
)

# generate the DGL dataset
dataset = CGCNNDataset(
    ids,
    cidxs,
    structures,
    energies,
    featurizer,
    name="cluster_graph",
    save_cache=True,
    save_dir="graph_dataset",
)

# get the edge feature dimension
in_edge_dim = dataset[0][2].edata["edge_attr"].shape[-1]

# split the dataset into train, validation, and test
train_dataset, val_dataset, test_dataset = split_dataset(
    dataset,
    frac_list=[args.train_ratio, args.val_ratio, args.test_ratio],
    shuffle=True,
    random_state=seed,
)

# generate the dataloaders
train_loader = GraphDataLoader(
    train_dataset,
    batch_size=args.b,
    num_workers=args.j,
    shuffle=True,
    drop_last=False,
)
val_loader = GraphDataLoader(
    val_dataset,
    batch_size=args.b,
    num_workers=args.j,
    shuffle=False,
    drop_last=False,
)
test_loader = GraphDataLoader(
    test_dataset,
    batch_size=args.b,
    num_workers=args.j,
    shuffle=False,
    drop_last=False,
)

# set up the model
model = CGCNN(
    in_edge_dim=in_edge_dim,
    n_tasks=1,
    tmax=args.epochs,
    **vars(args),
)

# set up the learning rate monitor and checkpoint callback
lr_monitor = LearningRateMonitor(logging_interval="epoch")
checkpoint_callback = ModelCheckpoint(
    monitor="val_mae",
    filename="{epoch:02d}-{step:02d}-{val_mae:.2f}",
    save_top_k=3,
    mode="min",
)
# set up the logger
logger = TensorBoardLogger("logs", name=args.exp_name, default_hp_metric=False)

# set up the trainer
trainer = pl.Trainer(
    max_epochs=args.epochs,
    accelerator=device,
    logger=logger,
    callbacks=[lr_monitor, checkpoint_callback],
)

# train the model
trainer.fit(model, train_loader, val_loader)

# test the model
trainer.test(dataloaders=val_loader)

print("Total time taken: %s" % (time.time() - start_time))
