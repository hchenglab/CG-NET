#!/usr/bin/env python
"""CG-NET Training Module - Modular training pipeline components."""

# Import the new modular trainer
from .trainer import CGNETTrainer

# Import individual managers for advanced usage
from .config_manager import ConfigManager
from .data_manager import DataManager
from .model_manager import ModelManager
from .training_manager import TrainingManager
from .slurm_manager import SlurmManager

__all__ = [
    'CGNETTrainer',
    'ConfigManager',
    'DataManager',
    'ModelManager',
    'TrainingManager',
    'SlurmManager'
] 