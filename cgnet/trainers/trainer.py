#!/usr/bin/env python
"""CG-NET Trainer - Main orchestrator for the modular training pipeline.

This module provides the main CGNETTrainer class that orchestrates the entire
machine learning pipeline including data preparation, model training, testing,
and prediction. It integrates with various managers to provide a clean and
modular architecture.
"""

import os
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import logging

from torch_geometric.loader import DataLoader
import pytorch_lightning as pl

from .config_manager import ConfigManager
from .data_manager import DataManager
from .model_manager import ModelManager
from .training_manager import TrainingManager
from .slurm_manager import SlurmManager
from ..utils import CGNETDataset

# Configure logging
logger = logging.getLogger(__name__)


class PipelineMode(Enum):
    """Enumeration of available pipeline execution modes."""
    ALL = "all"
    DATA = "data" 
    TRAIN = "train"
    TEST = "test"
    PREDICT = "predict"


class TrainerState:
    """Manages the internal state of the trainer."""
    
    def __init__(self):
        self.dataset: Optional[CGNETDataset] = None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None
        self.prediction_loader: Optional[DataLoader] = None
        self.model: Optional[Any] = None
        self.trainer: Optional[pl.Trainer] = None
    
    def is_dataset_ready(self) -> bool:
        """Check if dataset is ready."""
        return self.dataset is not None
    
    def are_dataloaders_ready(self) -> bool:
        """Check if all dataloaders are ready."""
        return all([
            self.train_loader is not None,
            self.val_loader is not None,
            self.test_loader is not None
        ])
    
    def is_prediction_loader_ready(self) -> bool:
        """Check if prediction loader is ready."""
        return self.prediction_loader is not None
    
    def is_model_trained(self) -> bool:
        """Check if model is trained."""
        return self.model is not None
    
    def is_trainer_ready(self) -> bool:
        """Check if trainer is ready."""
        return self.trainer is not None
    
    def reset(self) -> None:
        """Reset all state to None."""
        self.dataset = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.prediction_loader = None
        self.model = None
        self.trainer = None


class CGNETTrainer:
    """Main orchestrator for CG-NET training pipeline with modular architecture.
    
    This class provides a high-level interface for the complete machine learning
    workflow including data preparation, model training, evaluation, and prediction.
    It integrates with specialized managers for different aspects of the pipeline.
    
    Attributes:
        config_path (str): Path to the configuration file
        config (Dict[str, Any]): Loaded configuration
        data_manager (DataManager): Handles data operations
        model_manager (ModelManager): Handles model operations
        training_manager (TrainingManager): Handles training operations
        slurm_manager (Optional[SlurmManager]): Handles SLURM operations
        state (TrainerState): Internal state management
    """
    
    def __init__(self, config_path: str = "config.yml"):
        """Initialize trainer with configuration file.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValidationError: If config validation fails
        """
        self.config_path = config_path
        self.config = ConfigManager.load_config(config_path, silent_validation=True)
        
        # Initialize managers
        self._initialize_managers()
        
        # Initialize state
        self.state = TrainerState()
        
        logger.info(f"CGNETTrainer initialized with config: {config_path}")
    
    def _initialize_managers(self) -> None:
        """Initialize all manager instances."""
        self.data_manager = DataManager(self.config)
        self.model_manager = ModelManager(self.config)
        self.training_manager = TrainingManager(self.config, self.model_manager)
        
        # Only initialize SLURM manager if SLURM is enabled
        if self._is_slurm_enabled():
            self.slurm_manager = SlurmManager(self.config)
            logger.info("SLURM manager initialized")
        else:
            self.slurm_manager = None
            logger.debug("SLURM disabled, skipping SLURM manager initialization")
    
    def _is_slurm_enabled(self) -> bool:
        """Check if SLURM is enabled in configuration."""
        return self.config.get('slurm', {}).get('use_slurm', False)
    
    def _is_cross_validation_enabled(self) -> bool:
        """Check if cross-validation is enabled in configuration."""
        return self.config.get('cross_validation', {}).get('enabled', False)
    
    def _validate_mode(self, mode: str) -> PipelineMode:
        """Validate and convert mode string to PipelineMode enum.
        
        Args:
            mode: Mode string to validate
            
        Returns:
            PipelineMode enum value
            
        Raises:
            ValueError: If mode is not valid
        """
        try:
            return PipelineMode(mode)
        except ValueError:
            valid_modes = [m.value for m in PipelineMode]
            raise ValueError(f"Unknown mode: {mode}. Must be one of: {valid_modes}")
    
    def _ensure_dataset_ready(self) -> CGNETDataset:
        """Ensure dataset is ready, generating if necessary.
        
        Returns:
            The prepared dataset
        """
        if not self.state.is_dataset_ready():
            logger.info("Dataset not ready, generating...")
            self.generate_dataset()
        return self.state.dataset
    
    def _ensure_dataloaders_ready(self, dataset: Optional[CGNETDataset] = None) -> None:
        """Ensure all dataloaders are ready, preparing if necessary.
        
        Args:
            dataset: Optional dataset to use for preparation
        """
        if not self.state.are_dataloaders_ready():
            logger.info("Dataloaders not ready, preparing...")
            self.prepare_all_dataloaders(dataset)
    
    def generate_dataset(self) -> CGNETDataset:
        """Generate dataset from raw data.
        
        Returns:
            Generated CGNETDataset
            
        Note:
            If dataset already exists, skips generation and returns existing dataset.
        """
        if self.state.is_dataset_ready():
            logger.info("Dataset already exists, skipping generation")
            return self.state.dataset
            
        logger.info("Generating dataset from raw data...")
        self.state.dataset = self.data_manager.generate_dataset()
        logger.info(f"Dataset generated with {len(self.state.dataset)} samples")
        
        return self.state.dataset
    
    def prepare_all_dataloaders(self, dataset: Optional[CGNETDataset] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare data loaders for training, validation, and testing.
        
        Args:
            dataset: Optional dataset to use. If None, uses existing dataset or generates new one
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
            
        Note:
            If dataloaders already exist, skips preparation and returns existing loaders.
        """
        if dataset is None:
            dataset = self._ensure_dataset_ready()
        
        if self.state.are_dataloaders_ready():
            logger.info("Data loaders already exist, skipping preparation")
            return self.state.train_loader, self.state.val_loader, self.state.test_loader
        
        logger.info("Preparing training, validation, and test dataloaders...")
        (self.state.train_loader, 
         self.state.val_loader, 
         self.state.test_loader) = self.data_manager.prepare_all_dataloaders(dataset)
        
        logger.info("All dataloaders prepared successfully")
        return self.state.train_loader, self.state.val_loader, self.state.test_loader
    
    def prepare_prediction_dataloader(self, 
                                    dataset: Optional[CGNETDataset] = None, 
                                    batch_size: Optional[int] = None) -> DataLoader:
        """Prepare a single dataloader containing all data for prediction.
        
        Args:
            dataset: Dataset to use. If None, uses existing dataset or generates new one
            batch_size: Batch size for prediction. If None, uses config batch_size
            
        Returns:
            DataLoader containing all data for prediction
        """
        if dataset is None:
            dataset = self._ensure_dataset_ready()
        
        logger.info("Preparing prediction dataloader...")
        self.state.prediction_loader = self.data_manager.prepare_prediction_dataloader(
            dataset, batch_size)
        
        logger.info(f"Prediction dataloader prepared with {len(dataset)} samples")
        return self.state.prediction_loader

    def prepare_kfold_splits(self, 
                           dataset: Optional[CGNETDataset] = None, 
                           reserve_test: bool = True) -> Tuple[List[Tuple[DataLoader, DataLoader]], Optional[DataLoader]]:
        """Prepare k-fold cross-validation splits.
        
        Args:
            dataset: Dataset to use. If None, uses existing dataset or generates new one
            reserve_test: Whether to reserve a portion of data for final testing
            
        Returns:
            Tuple of (fold_loaders, test_loader) where fold_loaders is a list of 
            (train, val) loader pairs and test_loader is optional final test set
        """
        if dataset is None:
            dataset = self._ensure_dataset_ready()
        
        logger.info("Preparing k-fold cross-validation splits...")
        fold_loaders, test_loader = self.data_manager.prepare_kfold_splits(dataset, reserve_test)
        logger.info(f"K-fold splits prepared ({len(fold_loaders)} folds)")
        
        return fold_loaders, test_loader
    
    def train_model(self, 
                   train_loader: Optional[DataLoader] = None, 
                   val_loader: Optional[DataLoader] = None) -> Tuple[Any, pl.Trainer]:
        """Train the model using specified or default dataloaders.
        
        Args:
            train_loader: Training dataloader. If None, uses prepared train_loader
            val_loader: Validation dataloader. If None, uses prepared val_loader
            
        Returns:
            Tuple of (trained_model, pytorch_lightning_trainer)
            
        Raises:
            ValueError: If required dataloaders are not available
        """
        # Use provided loaders or fall back to prepared ones
        train_loader = train_loader or self.state.train_loader
        val_loader = val_loader or self.state.val_loader
        
        if train_loader is None or val_loader is None:
            raise ValueError(
                "Data loaders must be prepared before training. "
                "Call prepare_all_dataloaders() first or provide loaders directly."
            )
        
        logger.info("Starting model training...")
        self.state.model, self.state.trainer = self.training_manager.train_model(
            train_loader, val_loader)
        
        logger.info("Model training completed successfully")
        return self.state.model, self.state.trainer
    
    def test_model(self, test_loader: Optional[DataLoader] = None) -> Dict[str, float]:
        """Test the model and return test metrics.
        
        Args:
            test_loader: Test dataloader. If None, uses prepared test_loader
            
        Returns:
            Dictionary containing test metrics
            
        Raises:
            ValueError: If test_loader is not available or model is not trained
        """
        test_loader = test_loader or self.state.test_loader
        
        if test_loader is None:
            raise ValueError(
                "Test loader must be prepared before testing. "
                "Call prepare_all_dataloaders() first or provide loader directly."
            )
        
        if not self.state.is_model_trained():
            raise ValueError(
                "Model must be trained before testing. "
                "Call train_model() first."
            )
        
        logger.info("Starting model testing...")
        test_results = self.training_manager.test_model(
            self.state.model, test_loader, self.state.trainer)
        
        logger.info("Model testing completed successfully")
        return test_results
    
    def predict_with_model(self, 
                          model_path: str, 
                          data_loader: Optional[DataLoader] = None) -> Any:
        """Make predictions using a trained model.
        
        Args:
            model_path: Path to the model checkpoint
            data_loader: Custom dataloader to use. If None, creates prediction loader
            
        Returns:
            Model predictions
            
        Raises:
            FileNotFoundError: If model checkpoint doesn't exist
        """
        if data_loader is None:
            data_loader = self.prepare_prediction_dataloader()
        
        dataset_size = (len(data_loader.dataset) 
                       if hasattr(data_loader, 'dataset') else 'Unknown')
        
        logger.info(f"Making predictions with model: {model_path}")
        logger.info(f"Prediction data size: {dataset_size} samples")
        
        model = self.model_manager.load_model_from_checkpoint(model_path)
        predictions = self.training_manager.predict_with_model(model, data_loader)
        
        logger.info("Predictions completed successfully")
        return predictions
    
    def run_kfold_training(self, dataset: Optional[CGNETDataset] = None) -> Dict[str, Any]:
        """Run k-fold cross-validation training.
        
        Args:
            dataset: Dataset to use. If None, uses existing dataset or generates new one
            
        Returns:
            Dictionary containing k-fold training results
        """
        if dataset is None:
            dataset = self._ensure_dataset_ready()
        
        logger.info("Starting k-fold cross-validation training...")
        
        # Prepare k-fold splits
        fold_loaders, test_loader = self.prepare_kfold_splits(dataset)
        
        # Run k-fold training
        results = self.training_manager.run_kfold_training(fold_loaders, test_loader)
        
        logger.info("K-fold cross-validation completed successfully")
        return results
    
    def run_pipeline(self, 
                    mode: str = "all", 
                    model_path: Optional[str] = None) -> Dict[str, Any]:
        """Run the complete training pipeline.
        
        Args:
            mode: Pipeline mode - one of: 'all', 'data', 'train', 'test', 'predict'
            model_path: Path to model checkpoint (required for prediction mode)
            
        Returns:
            Dictionary containing results from the pipeline execution
            
        Raises:
            ValueError: If mode is invalid or required parameters are missing
            
        Notes:
            - 'all': Complete ML workflow (data + train + test)
            - 'data': Data preparation only
            - 'train': Training only 
            - 'test': Testing only (requires trained model)
            - 'predict': Prediction only (requires model_path and data)
        """
        pipeline_mode = self._validate_mode(mode)
        
        results = {}
        logger.info(f"Starting pipeline in '{mode}' mode...")
        
        try:
            # Data preparation
            if pipeline_mode in [PipelineMode.ALL, PipelineMode.DATA]:
                dataset = self.generate_dataset()
                results['dataset_size'] = len(dataset)
                
                # Only prepare loaders if needed for training/testing and not using cross-validation
                if (pipeline_mode != PipelineMode.DATA and 
                    not self._is_cross_validation_enabled()):
                    self.prepare_all_dataloaders()
            
            # Training
            if pipeline_mode in [PipelineMode.ALL, PipelineMode.TRAIN]:
                self._run_training_phase(results)
            
            # Testing (evaluation on test set)
            if pipeline_mode in [PipelineMode.ALL, PipelineMode.TEST]:
                self._run_testing_phase(results, pipeline_mode)
            
            # Prediction (inference - separate from training workflow)
            if pipeline_mode == PipelineMode.PREDICT:
                self._run_prediction_phase(results, model_path)
            
            logger.info(f"Pipeline '{mode}' completed successfully!")
            
        except Exception as e:
            logger.error(f"Pipeline '{mode}' failed: {str(e)}")
            raise
        
        return results
    
    def _run_training_phase(self, results: Dict[str, Any]) -> None:
        """Execute the training phase of the pipeline.
        
        Args:
            results: Results dictionary to update
        """
        # Ensure data is prepared
        self._ensure_dataset_ready()
        
        if self._is_cross_validation_enabled():
            logger.info("Running k-fold cross-validation training...")
            results['training_results'] = self.run_kfold_training()
        else:
            logger.info("Running standard training...")
            self._ensure_dataloaders_ready()
            model, trainer = self.train_model()
            results['model'] = model
            results['trainer'] = trainer
    
    def _run_testing_phase(self, results: Dict[str, Any], mode: PipelineMode) -> None:
        """Execute the testing phase of the pipeline.
        
        Args:
            results: Results dictionary to update
            mode: Pipeline mode
            
        Raises:
            ValueError: If model is not available for testing
        """
        # Ensure model and data are ready
        if not self.state.is_model_trained() and mode == PipelineMode.TEST:
            raise ValueError(
                "No trained model found. Run training first or provide a model checkpoint."
            )
        
        # Skip testing if cross-validation is enabled (testing is handled within k-fold)
        if not self._is_cross_validation_enabled():
            self._ensure_dataloaders_ready()
            logger.info("Running model testing...")
            results['test_results'] = self.test_model()
    
    def _run_prediction_phase(self, results: Dict[str, Any], model_path: Optional[str]) -> None:
        """Execute the prediction phase of the pipeline.
        
        Args:
            results: Results dictionary to update
            model_path: Path to model checkpoint
            
        Raises:
            ValueError: If model_path is not provided or model doesn't exist
        """
        if model_path is None:
            model_path = self.model_manager.get_latest_checkpoint()
        
        if not model_path:
            raise ValueError(
                "Model checkpoint is required for prediction mode. "
                "Provide model_path or ensure a trained model exists."
            )
        
        # Ensure dataset is available
        dataset = self._ensure_dataset_ready()
        
        logger.info("Running model prediction...")
        predictions = self.predict_with_model(model_path)
        results['predictions'] = predictions
        results['prediction_data_size'] = len(dataset)
    
    def submit_pipeline(self, 
                       mode: str = "all", 
                       model_path: Optional[str] = None) -> Any:
        """Submit the pipeline to SLURM for execution.
        
        Args:
            mode: Pipeline mode to execute
            model_path: Optional model path for prediction mode
            
        Returns:
            SLURM job object
            
        Raises:
            RuntimeError: If SLURM is not enabled
        """
        if self.slurm_manager is None:
            raise RuntimeError(
                "SLURM is not enabled in configuration. "
                "Set 'slurm.use_slurm: true' to enable."
            )
        
        logger.info(f"Submitting pipeline to SLURM (mode: {mode})...")
        job = self.slurm_manager.submit_job(
            mode=mode,
            model_path=model_path,
            config_path=self.config_path
        )
        
        logger.info(f"Job submitted successfully: {job}")
        return job
    
    @staticmethod
    def monitor_jobs(jobs: List[Any]) -> Dict[str, Any]:
        """Monitor SLURM jobs status.
        
        Args:
            jobs: List of SLURM job objects to monitor
            
        Returns:
            Dictionary containing job status information
        """
        logger.info(f"Monitoring {len(jobs)} SLURM jobs...")
        return SlurmManager.monitor_jobs(jobs)
    
    @staticmethod
    def cancel_jobs(jobs: List[Any]) -> None:
        """Cancel SLURM jobs.
        
        Args:
            jobs: List of SLURM job objects to cancel
        """
        logger.info(f"Cancelling {len(jobs)} SLURM jobs...")
        SlurmManager.cancel_jobs(jobs)
        logger.info("Jobs cancelled successfully")
    
    def get_model_summary(self) -> str:
        """Get a summary of the model architecture.
        
        Returns:
            String representation of model architecture
        """
        return self.model_manager.get_model_summary()
    
    def get_pipeline_state(self) -> Dict[str, bool]:
        """Get the current state of the pipeline components.
        
        Returns:
            Dictionary indicating readiness status of pipeline components
        """
        return {
            'dataset_ready': self.state.is_dataset_ready(),
            'dataloaders_ready': self.state.are_dataloaders_ready(),
            'prediction_loader_ready': self.state.is_prediction_loader_ready(),
            'model_trained': self.state.is_model_trained(),
            'trainer_ready': self.state.is_trainer_ready()
        }
    
    def reset(self) -> None:
        """Reset the trainer state and all managers.
        
        This clears all cached data, models, and dataloaders, allowing for
        a fresh start of the pipeline.
        """
        logger.info("Resetting trainer state...")
        
        # Reset internal state
        self.state.reset()
        
        # Reset managers
        self.model_manager.reset()
        self.training_manager.reset()
        
        logger.info("Trainer state reset successfully")
    