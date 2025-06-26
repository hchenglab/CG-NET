#!/usr/bin/env python

"""
CG-NET: Cluster Graph Neural Network

A PyTorch implementation of Cluster Graph Neural Network for materials property prediction.
"""

# Import main classes from subpackages
from .models import CGNET, CGNETLayer
from .utils import CGNETFeatureizer, CGNETDataset, GraphData
from .trainers import CGNETTrainer

# Version information
__version__ = "0.1.0"

# Expose main API
__all__ = [
    "CGNET", 
    "CGNETLayer",
    "CGNETFeatureizer",
    "CGNETDataset", 
    "GraphData",
    "CGNETTrainer",
    "__version__"
] 