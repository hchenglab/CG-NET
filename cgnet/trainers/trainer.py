#!/usr/bin/env python
"""CG-NET Trainer - Main orchestrator for the modular training pipeline."""

import os
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import pandas as pd
import numpy as np

import torch
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl

from .config_manager import ConfigManager
from .data_manager import DataManager
from .model_manager import ModelManager
from .training_manager import TrainingManager
from .slurm_manager import SlurmManager
from ..utils import CGNETDataset


class CGNETTrainer:
    """Main orchestrator for CG-NET training pipeline with modular architecture."""
    
    def __init__(self, config_path: str = "config.yml"):
        """Initialize trainer with configuration file."""
        self.config_path = config_path
        self.config = ConfigManager.load_config(config_path)
        
        # Initialize managers
        self.data_manager = DataManager(self.config)
        self.model_manager = ModelManager(self.config)
        self.training_manager = TrainingManager(self.config, self.model_manager)
        
        # Only initialize SLURM manager if SLURM is enabled
        if self.config.get('slurm', {}).get('use_slurm', False):
            self.slurm_manager = SlurmManager(self.config)
        else:
            self.slurm_manager = None
        
        # Initialize state
        self.dataset = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.prediction_loader = None
        self.model = None
        self.trainer = None
    
    def generate_dataset(self) -> CGNETDataset:
        """Generate dataset from raw data."""
        if self.dataset is None:
            print("Generating dataset...")
            self.dataset = self.data_manager.generate_dataset()
            print(f"Dataset generated with {len(self.dataset)} samples")
        else:
            print("Dataset already exists, skipping generation")
        return self.dataset
    
    def prepare_all_dataloaders(self, dataset: Optional[CGNETDataset] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare data loaders for training, validation, and testing."""
        if dataset is None:
            dataset = self.dataset if self.dataset is not None else self.generate_dataset()
        
        if self.train_loader is None or self.val_loader is None or self.test_loader is None:
            print("Preparing data loaders...")
            self.train_loader, self.val_loader, self.test_loader = self.data_manager.prepare_all_dataloaders(dataset)
            print("Data loaders prepared")
        else:
            print("Data loaders already exist, skipping preparation")
        
        return self.train_loader, self.val_loader, self.test_loader
    
    def prepare_prediction_dataloader(self, dataset: Optional[CGNETDataset] = None, batch_size: Optional[int] = None) -> DataLoader:
        """
        Prepare a single dataloader containing all data for prediction.
        
        Parameters
        ----------
        dataset : CGNETDataset, optional
            Dataset to use. If None, uses self.dataset or generates it.
        batch_size : int, optional
            Batch size for prediction. If None, uses config batch_size.
            
        Returns
        -------
        DataLoader
            A single dataloader containing all data for prediction.
        """
        if dataset is None:
            dataset = self.dataset if self.dataset is not None else self.generate_dataset()
        
        self.prediction_loader = self.data_manager.prepare_prediction_dataloader(dataset, batch_size)
        
        return self.prediction_loader

    def prepare_kfold_splits(self, dataset: Optional[CGNETDataset] = None, 
                           reserve_test: bool = True) -> Tuple[List[Tuple[DataLoader, DataLoader]], Optional[DataLoader]]:
        """Prepare k-fold cross-validation splits."""
        if dataset is None:
            dataset = self.dataset if self.dataset is not None else self.generate_dataset()
        
        print("Preparing k-fold splits...")
        fold_loaders, test_loader = self.data_manager.prepare_kfold_splits(dataset, reserve_test)
        print(f"K-fold splits prepared ({len(fold_loaders)} folds)")
        
        return fold_loaders, test_loader
    
    def train_model(self, train_loader=None, val_loader=None) -> Tuple[Any, pl.Trainer]:
        """Train the model."""
        if train_loader is None:
            train_loader = self.train_loader
        if val_loader is None:
            val_loader = self.val_loader
            
        if train_loader is None or val_loader is None:
            raise ValueError("Data loaders must be prepared before training. Call prepare_all_dataloaders() first.")
        
        print("Starting model training...")
        # Train model using training manager
        self.model, self.trainer = self.training_manager.train_model(
            train_loader, val_loader)
        print("Model training completed")
        
        return self.model, self.trainer
    
    def test_model(self, test_loader=None) -> Dict[str, float]:
        """Test the model and return test metrics."""
        if test_loader is None:
            test_loader = self.test_loader
            
        if test_loader is None:
            raise ValueError("Test loader must be prepared before testing. Call prepare_all_dataloaders() first.")
        if self.model is None:
            raise ValueError("Model must be trained before testing. Call train_model() first.")
        
        print("Testing model...")
        test_results = self.training_manager.test_model(self.model, test_loader, self.trainer)
        print("Model testing completed")
        
        return test_results
    
    def predict_with_model(self, model_path: str, data_loader=None) -> Any:
        """
        Make predictions using a trained model.
        
        Parameters
        ----------
        model_path : str
            Path to the model checkpoint.
        data_loader : DataLoader, optional
            Custom dataloader to use. If None and use_all_data=True, 
            creates a prediction loader with all data.
            
        Returns
        -------
        predictions
            Model predictions.
        """
        if data_loader is None:
            data_loader = self.prepare_prediction_dataloader()
        
        print(f"Making predictions with model: {model_path}")
        print(f"Prediction data size: {len(data_loader.dataset) if hasattr(data_loader, 'dataset') else 'Unknown'} samples")
        
        model = self.model_manager.load_model_from_checkpoint(model_path)
        predictions = self.training_manager.predict_with_model(model, data_loader)
        print("Predictions completed")
        
        return predictions
    
    def run_kfold_training(self, dataset: Optional[CGNETDataset] = None) -> Dict[str, Any]:
        """Run k-fold cross-validation training."""
        if dataset is None:
            dataset = self.dataset if self.dataset is not None else self.generate_dataset()
        
        print("Starting k-fold cross-validation...")
        # Prepare k-fold splits
        fold_loaders, test_loader = self.prepare_kfold_splits(dataset)
        
        # Run k-fold training
        results = self.training_manager.run_kfold_training(fold_loaders, test_loader)
        print("K-fold cross-validation completed")
        
        return results
    
    def run_pipeline(self, mode: str = "all", model_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the complete training pipeline.
        
        Parameters
        ----------
        mode : str
            Pipeline mode:
            - "all": Complete ML workflow (data + train + test)
            - "data": Data preparation only
            - "train": Training only 
            - "test": Testing only (requires trained model)
            - "predict": Prediction only (requires model_path and data)
        model_path : str, optional
            Path to model checkpoint for prediction mode
            
        Returns
        -------
        dict
            Results from the pipeline execution
            
        Notes
        -----
        The 'predict' mode is designed for inference on new data and is separate 
        from the standard ML workflow. By default, it uses all available data for 
        prediction. Set predict_on_all_data=False to use only test data.
        """
        if mode not in ["all", "data", "train", "test", "predict"]:
            raise ValueError(f"Unknown mode: {mode}. Must be one of: all, data, train, test, predict")
        
        results = {}
        print(f"Starting pipeline in '{mode}' mode...")
        
        # Data preparation
        if mode in ["all", "data"]:
            print("=== DATA PREPARATION ===")
            dataset = self.generate_dataset()
            results['dataset_size'] = len(dataset)
            
            if mode != "data":  # Only prepare loaders if we need them for training/testing
                self.prepare_all_dataloaders()
        
        # Training
        if mode in ["all", "train"]:
            print("=== TRAINING ===")
            # Ensure data is prepared
            if self.dataset is None:
                self.generate_dataset()
            if self.train_loader is None:
                self.prepare_all_dataloaders()
                
            if self.config.get('cross_validation', {}).get('enabled', False):
                results['training_results'] = self.run_kfold_training()
            else:
                model, trainer = self.train_model()
                results['model'] = model
                results['trainer'] = trainer
        
        # Testing (evaluation on test set)
        if mode in ["all", "test"]:
            print("=== TESTING & EVALUATION ===")
            # Ensure model and data are ready
            if self.model is None and mode == "test":
                raise ValueError("No trained model found. Run training first or provide a model checkpoint.")
            if self.test_loader is None:
                if self.dataset is None:
                    self.generate_dataset()
                self.prepare_all_dataloaders()
                
            results['test_results'] = self.test_model()
        
        # Prediction (inference - separate from training workflow)
        if mode == "predict":
            print("=== PREDICTION (INFERENCE) ===")
            if model_path is None:
                model_path = self.model_manager.get_latest_checkpoint()
            if not model_path:
                raise ValueError("Model checkpoint is required for prediction mode. Provide model_path or ensure a trained model exists.")
            
            # Ensure dataset is available
            if self.dataset is None:
                self.generate_dataset()
                
            predictions = self.predict_with_model(model_path)
            results['predictions'] = predictions
            results['prediction_data_size'] = len(self.dataset)
        
        print(f"Pipeline '{mode}' completed successfully!")
        return results
    
    def submit_pipeline(self, mode: str = "all", model_path: Optional[str] = None) -> Any:
        """Submit the pipeline to SLURM."""
        if self.slurm_manager is None:
            raise RuntimeError("SLURM is not enabled in configuration. Set 'slurm.use_slurm: true' to enable.")
        
        print(f"Submitting pipeline to SLURM (mode: {mode})...")
        job = self.slurm_manager.submit_job(
            mode=mode,
            model_path=model_path,
            config_path=self.config_path
        )
        return job
    

    
    @staticmethod
    def monitor_jobs(jobs: List[Any]) -> Dict[str, Any]:
        """Monitor SLURM jobs."""
        return SlurmManager.monitor_jobs(jobs)
    
    @staticmethod
    def cancel_jobs(jobs: List[Any]) -> None:
        """Cancel SLURM jobs."""
        print(f"Cancelling {len(jobs)} SLURM jobs...")
        SlurmManager.cancel_jobs(jobs)
        print("Jobs cancelled")
    
    def get_model_summary(self) -> str:
        """Get a summary of the model architecture."""
        return self.model_manager.get_model_summary()
    
    def get_pipeline_state(self) -> Dict[str, bool]:
        """Get the current state of the pipeline components."""
        return {
            'dataset_ready': self.dataset is not None,
            'dataloaders_ready': all([
                self.train_loader is not None,
                self.val_loader is not None,
                self.test_loader is not None
            ]),
            'prediction_loader_ready': self.prediction_loader is not None,
            'model_trained': self.model is not None,
            'trainer_ready': self.trainer is not None
        }
    
    def reset(self) -> None:
        """Reset the trainer state."""
        print("Resetting trainer state...")
        self.dataset = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.prediction_loader = None
        self.model = None
        self.trainer = None
        self.model_manager.reset()
        self.training_manager.reset()
        print("Trainer state reset")
    