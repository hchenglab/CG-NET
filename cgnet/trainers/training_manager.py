#!/usr/bin/env python
"""Training management for CG-NET training pipeline.

This module provides comprehensive training management capabilities including:
- Model training with PyTorch Lightning integration
- K-fold cross-validation training and evaluation
- Model testing and prediction workflows
- Callback and logger configuration
- Metrics aggregation and result analysis
- Training state management and persistence

The TrainingManager class serves as the central hub for all training-related
operations in the CG-NET pipeline.
"""

import os
import pickle
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field
import warnings

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch_geometric.loader import DataLoader

from .config_manager import ConfigManager
from .model_manager import ModelManager

# Configure logging
logger = logging.getLogger(__name__)

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning")


class TrainingError(Exception):
    """Base exception for training-related errors."""
    pass


class TrainingConfigurationError(TrainingError):
    """Exception raised for training configuration errors."""
    pass


class ModelTrainingError(TrainingError):
    """Exception raised for model training errors."""
    pass


class ValidationError(TrainingError):
    """Exception raised for validation errors."""
    pass


class TaskType(Enum):
    """Enumeration of supported task types."""
    REGRESSION = "regression"
    CLASSIFICATION = "classification"


class MetricMode(Enum):
    """Enumeration of metric optimization modes."""
    MINIMIZE = "min"
    MAXIMIZE = "max"


@dataclass
class TrainingConfig:
    """Data class for training configuration."""
    epochs: int
    devices: Union[int, List[int]] = 1
    strategy: Optional[str] = None
    accelerator: str = "auto"
    precision: Union[int, str] = 32
    enable_progress_bar: bool = True
    enable_model_summary: bool = True
    gradient_clip_val: Optional[float] = None
    accumulate_grad_batches: int = 1
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create TrainingConfig from dictionary."""
        return cls(
            epochs=config_dict.get('epochs', 100),
            devices=config_dict.get('devices', 1),
            strategy=config_dict.get('strategy'),
            accelerator=config_dict.get('accelerator', 'auto'),
            precision=config_dict.get('precision', 32),
            enable_progress_bar=config_dict.get('enable_progress_bar', True),
            enable_model_summary=config_dict.get('enable_model_summary', True),
            gradient_clip_val=config_dict.get('gradient_clip_val'),
            accumulate_grad_batches=config_dict.get('accumulate_grad_batches', 1)
        )


@dataclass
class EarlyStoppingConfig:
    """Data class for early stopping configuration."""
    enabled: bool = False
    monitor: str = "val_loss"
    patience: int = 10
    mode: str = "min"
    min_delta: float = 0.0
    verbose: bool = True
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'EarlyStoppingConfig':
        """Create EarlyStoppingConfig from dictionary."""
        return cls(
            enabled=config_dict.get('enabled', False),
            monitor=config_dict.get('monitor', 'val_loss'),
            patience=config_dict.get('patience', 10),
            mode=config_dict.get('mode', 'min'),
            min_delta=config_dict.get('min_delta', 0.0),
            verbose=config_dict.get('verbose', True)
        )


@dataclass
class LoggingConfig:
    """Data class for logging configuration."""
    log_dir: str = "logs"
    monitor: str = "val_loss"
    mode: str = "min"
    save_top_k: int = 1
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'LoggingConfig':
        """Create LoggingConfig from dictionary."""
        return cls(
            log_dir=config_dict.get('log_dir', 'logs'),
            monitor=config_dict.get('monitor', 'val_loss'),
            mode=config_dict.get('mode', 'min'),
            save_top_k=config_dict.get('save_top_k', 1)
        )


@dataclass
class CrossValidationConfig:
    """Data class for cross-validation configuration."""
    enabled: bool = False
    k_folds: int = 5
    save_fold_results: bool = True
    ensemble_prediction: bool = True
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CrossValidationConfig':
        """Create CrossValidationConfig from dictionary."""
        return cls(
            enabled=config_dict.get('enabled', False),
            k_folds=config_dict.get('k_folds', 5),
            save_fold_results=config_dict.get('save_fold_results', True),
            ensemble_prediction=config_dict.get('ensemble_prediction', True)
        )


@dataclass
class FoldResult:
    """Data class for storing results from a single fold."""
    fold_idx: int
    metrics: Dict[str, float]
    model_path: Optional[str] = None
    trainer_log_dir: Optional[str] = None
    training_time: Optional[float] = None
    validation_time: Optional[float] = None
    
    @property
    def total_time(self) -> Optional[float]:
        """Get total time for this fold."""
        if self.training_time and self.validation_time:
            return self.training_time + self.validation_time
        return None


@dataclass
class MetricStatistics:
    """Data class for metric statistics across folds."""
    mean: float
    std: float
    min: float
    max: float
    values: List[float]
    
    @property
    def confidence_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval."""
        import scipy.stats as stats
        n = len(self.values)
        if n < 2:
            return self.mean, self.mean
        
        t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
        margin = t_value * (self.std / np.sqrt(n))
        return self.mean - margin, self.mean + margin


class TrainingConstants:
    """Constants used throughout training management."""
    
    # Default values
    DEFAULT_EPOCHS = 100
    DEFAULT_PATIENCE = 10
    DEFAULT_DEVICES = 1
    DEFAULT_PRECISION = 32
    DEFAULT_LOG_DIR = "logs"
    DEFAULT_MONITOR_METRIC = "val_loss"
    DEFAULT_MONITOR_MODE = "min"
    
    # File names
    KFOLD_RESULTS_FILE = "kfold_results.pkl"
    CONFIG_FILE = "config.yml"
    FOLD_CHECKPOINT_TEMPLATE = "fold_{}_final.ckpt"
    
    # Formatting
    SEPARATOR_LONG = "=" * 60
    SEPARATOR_MEDIUM = "=" * 50
    SEPARATOR_SHORT = "-" * 30


class ConfigValidator:
    """Handles training configuration validation."""
    
    @staticmethod
    def validate_training_config(config: Dict[str, Any]) -> None:
        """
        Validate training configuration.
        
        Args:
            config: Configuration dictionary to validate
            
        Raises:
            TrainingConfigurationError: If configuration is invalid
        """
        # Check required sections
        required_sections = ['training', 'logging', 'model', 'experiment']
        for section in required_sections:
            if section not in config:
                raise TrainingConfigurationError(f"Missing required section '{section}' in configuration")
        
        # Validate training parameters
        ConfigValidator._validate_training_parameters(config['training'])
        
        # Validate logging parameters
        ConfigValidator._validate_logging_parameters(config['logging'])
        
        # Validate model parameters
        ConfigValidator._validate_model_parameters(config['model'])
    
    @staticmethod
    def _validate_training_parameters(training_config: Dict[str, Any]) -> None:
        """Validate training parameters."""
        # Validate epochs
        epochs = training_config.get('epochs', TrainingConstants.DEFAULT_EPOCHS)
        if not isinstance(epochs, int) or epochs <= 0:
            raise TrainingConfigurationError(f"Epochs must be a positive integer, got {epochs}")
        
        # Validate devices
        devices = training_config.get('devices', TrainingConstants.DEFAULT_DEVICES)
        if not isinstance(devices, (int, list)) or (isinstance(devices, int) and devices <= 0):
            raise TrainingConfigurationError(f"Devices must be a positive integer or list, got {devices}")
        
        # Validate precision
        precision = training_config.get('precision', TrainingConstants.DEFAULT_PRECISION)
        if precision not in [16, 32, 64, "16", "32", "64", "bf16"]:
            raise TrainingConfigurationError(f"Invalid precision: {precision}")
    
    @staticmethod
    def _validate_logging_parameters(logging_config: Dict[str, Any]) -> None:
        """Validate logging parameters."""
        # Validate monitor mode
        mode = logging_config.get('mode', TrainingConstants.DEFAULT_MONITOR_MODE)
        if mode not in ['min', 'max']:
            raise TrainingConfigurationError(f"Monitor mode must be 'min' or 'max', got {mode}")
        
        # Validate save_top_k
        save_top_k = logging_config.get('save_top_k', 1)
        if not isinstance(save_top_k, int) or save_top_k < -1:
            raise TrainingConfigurationError(f"save_top_k must be -1 or a non-negative integer, got {save_top_k}")
    
    @staticmethod
    def _validate_model_parameters(model_config: Dict[str, Any]) -> None:
        """Validate model parameters."""
        # Validate task type
        task = model_config.get('task')
        if task not in [TaskType.REGRESSION.value, TaskType.CLASSIFICATION.value]:
            raise TrainingConfigurationError(f"Task must be 'regression' or 'classification', got {task}")


class CallbackManager:
    """Manages PyTorch Lightning callbacks."""
    
    @staticmethod
    def create_callbacks(training_config: TrainingConfig,
                        logging_config: LoggingConfig,
                        early_stopping_config: EarlyStoppingConfig) -> List[pl.Callback]:
        """
        Create and configure training callbacks.
        
        Args:
            training_config: Training configuration
            logging_config: Logging configuration
            early_stopping_config: Early stopping configuration
            
        Returns:
            List of configured callbacks
        """
        callbacks = []
        
        # Learning rate monitor
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)
        
        # Model checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            monitor=logging_config.monitor,
            filename="{epoch:02d}-{step:02d}-{" + logging_config.monitor + ":.2f}",
            save_top_k=logging_config.save_top_k,
            mode=logging_config.mode,
            save_last=True,
            verbose=False
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping callback
        if early_stopping_config.enabled:
            early_stopping = EarlyStopping(
                monitor=early_stopping_config.monitor,
                patience=early_stopping_config.patience,
                mode=early_stopping_config.mode,
                min_delta=early_stopping_config.min_delta,
                verbose=early_stopping_config.verbose
            )
            callbacks.append(early_stopping)
            logger.info(f"Early stopping enabled: monitor={early_stopping_config.monitor}, "
                       f"patience={early_stopping_config.patience}")
        
        return callbacks


class MetricsCalculator:
    """Handles metrics calculation for different task types."""
    
    @staticmethod
    def calculate_regression_metrics(predictions: torch.Tensor, 
                                   targets: torch.Tensor) -> Dict[str, float]:
        """Calculate regression metrics."""
        mse = torch.nn.functional.mse_loss(predictions, targets).item()
        mae = torch.nn.functional.l1_loss(predictions, targets).item()
        rmse = np.sqrt(mse)
        
        # Calculate R-squared
        ss_res = torch.sum((targets - predictions) ** 2).item()
        ss_tot = torch.sum((targets - torch.mean(targets)) ** 2).item()
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
    
    @staticmethod
    def calculate_classification_metrics(predictions: torch.Tensor,
                                       targets: torch.Tensor) -> Dict[str, float]:
        """Calculate classification metrics."""
        # Handle multi-class predictions
        if predictions.dim() > 1 and predictions.size(1) > 1:
            pred_classes = torch.argmax(predictions, dim=1)
        else:
            pred_classes = (predictions > 0.5).long().squeeze()
        
        # Handle multi-class targets
        if targets.dim() > 1 and targets.size(1) > 1:
            true_classes = torch.argmax(targets, dim=1)
        else:
            true_classes = targets.long().squeeze()
        
        # Calculate accuracy
        correct = (pred_classes == true_classes).float()
        accuracy = correct.mean().item()
        
        # Calculate precision, recall, F1 for binary classification
        metrics = {'accuracy': accuracy}
        
        if len(torch.unique(true_classes)) == 2:  # Binary classification
            tp = ((pred_classes == 1) & (true_classes == 1)).float().sum().item()
            fp = ((pred_classes == 1) & (true_classes == 0)).float().sum().item()
            fn = ((pred_classes == 0) & (true_classes == 1)).float().sum().item()
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
            
            metrics.update({
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
        
        return metrics


class TrainingManager:
    """Manages training operations for CG-NET including k-fold cross-validation.
    
    This class provides a comprehensive interface for training management including:
    - Model training with PyTorch Lightning integration
    - K-fold cross-validation training and evaluation
    - Model testing and prediction workflows
    - Callback and logger configuration
    - Metrics aggregation and result analysis
    - Training state management and persistence
    
    Attributes:
        config (Dict[str, Any]): Configuration dictionary
        model_manager (ModelManager): Model manager instance
        trainer (Optional[pl.Trainer]): Current PyTorch Lightning trainer
        training_config (TrainingConfig): Training configuration
        logging_config (LoggingConfig): Logging configuration
        early_stopping_config (EarlyStoppingConfig): Early stopping configuration
        cv_config (CrossValidationConfig): Cross-validation configuration
    """
    
    def __init__(self, config: Dict[str, Any], model_manager: ModelManager):
        """
        Initialize TrainingManager.
        
        Args:
            config: Configuration dictionary
            model_manager: Model manager instance
            
        Raises:
            TrainingConfigurationError: If configuration is invalid
        """
        self.config = config
        self.model_manager = model_manager
        self.trainer: Optional[pl.Trainer] = None
        
        # Validate configuration
        ConfigValidator.validate_training_config(config)
        
        # Parse configuration into structured objects
        self.training_config = TrainingConfig.from_dict(config.get('training', {}))
        self.logging_config = LoggingConfig.from_dict(config.get('logging', {}))
        self.early_stopping_config = EarlyStoppingConfig.from_dict(
            config.get('training', {}).get('early_stopping', {})
        )
        self.cv_config = CrossValidationConfig.from_dict(
            config.get('cross_validation', {})
        )
        
        logger.info("TrainingManager initialized successfully")
    
    def setup_trainer(self, 
                     experiment_name: Optional[str] = None, 
                     fold_info: Optional[str] = None,
                     enable_logging: bool = True) -> pl.Trainer:
        """
        Setup PyTorch Lightning trainer with callbacks and logger.
        
        Args:
            experiment_name: Name of the experiment
            fold_info: Information about the current fold (for k-fold CV)
            enable_logging: Whether to enable logging
            
        Returns:
            Configured PyTorch Lightning trainer
            
        Raises:
            TrainingConfigurationError: If trainer setup fails
        """
        logger.info("Setting up PyTorch Lightning trainer...")
        
        try:
            # Create callbacks
            callbacks = CallbackManager.create_callbacks(
                self.training_config,
                self.logging_config,
                self.early_stopping_config
            )
            
            # Setup logger
            trainer_logger = None
            if enable_logging:
                trainer_logger = self._setup_logger(experiment_name, fold_info)
            
            # Determine device configuration
            device_type = self._get_device_type()
            
            # Build trainer arguments
            trainer_kwargs = {
                'max_epochs': self.training_config.epochs,
                'accelerator': device_type,
                'devices': self.training_config.devices,
                'precision': self.training_config.precision,
                'logger': trainer_logger,
                'callbacks': callbacks,
                'enable_progress_bar': self.training_config.enable_progress_bar,
                'enable_model_summary': self.training_config.enable_model_summary,
            }
            
            # Add optional parameters
            if self.training_config.strategy:
                trainer_kwargs['strategy'] = self.training_config.strategy
            
            if self.training_config.gradient_clip_val:
                trainer_kwargs['gradient_clip_val'] = self.training_config.gradient_clip_val
            
            if self.training_config.accumulate_grad_batches > 1:
                trainer_kwargs['accumulate_grad_batches'] = self.training_config.accumulate_grad_batches
            
            # Create trainer
            trainer = pl.Trainer(**trainer_kwargs)
            self.trainer = trainer
            
            logger.info(f"Trainer setup completed: device={device_type}, "
                       f"devices={self.training_config.devices}, "
                       f"precision={self.training_config.precision}")
            
            return trainer
            
        except Exception as e:
            raise TrainingConfigurationError(f"Failed to setup trainer: {str(e)}") from e
    
    def _setup_logger(self, 
                     experiment_name: Optional[str] = None, 
                     fold_info: Optional[str] = None) -> TensorBoardLogger:
        """Setup TensorBoard logger."""
        exp_name = experiment_name or self.config['experiment']['name']
        if fold_info:
            exp_name = f"{exp_name}_{fold_info}"
        
        trainer_logger = TensorBoardLogger(
            self.logging_config.log_dir,
            name=exp_name,
            default_hp_metric=False
        )
        
        # Save config to log directory
        if trainer_logger.log_dir:
            config_path = Path(trainer_logger.log_dir) / TrainingConstants.CONFIG_FILE
            ConfigManager.save_config(self.config, str(config_path))
            logger.debug(f"Configuration saved to: {config_path}")
        
        return trainer_logger
    
    def _get_device_type(self) -> str:
        """Determine the appropriate device type."""
        if self.training_config.accelerator != "auto":
            return self.training_config.accelerator
        
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def train_model(self, 
                   train_loader: DataLoader, 
                   val_loader: DataLoader,
                   experiment_name: Optional[str] = None, 
                   fold_info: Optional[str] = None) -> Tuple[Any, pl.Trainer]:
        """
        Train the CG-NET model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            experiment_name: Name of the experiment
            fold_info: Information about the current fold
            
        Returns:
            Tuple of (trained_model, trainer)
            
        Raises:
            ModelTrainingError: If training fails
        """
        logger.info("Starting model training...")
        self._print_training_header()
        
        try:
            # Get edge dimension from sample data
            in_edge_dim = self._get_edge_dimension(train_loader)
            
            # Create model and trainer
            model = self.model_manager.create_model(in_edge_dim)
            trainer = self.setup_trainer(experiment_name, fold_info)
            
            device_type = self._get_device_type()
            logger.info(f"Training on device: {device_type}")
            
            # Record training start time
            start_time = time.time()
            
            # Start training
            trainer.fit(model, train_loader, val_loader)
            
            # Record training end time
            training_time = time.time() - start_time
            logger.info(f"Training completed in {training_time:.2f} seconds")
            
            return model, trainer
            
        except Exception as e:
            raise ModelTrainingError(f"Model training failed: {str(e)}") from e
    
    def _get_edge_dimension(self, data_loader: DataLoader) -> int:
        """Get edge dimension from sample data."""
        try:
            # Get sample data
            if hasattr(data_loader.dataset, 'dataset'):
                sample_data = data_loader.dataset.dataset[0]
            else:
                sample_data = data_loader.dataset[0]
            
            # Get edge dimension
            if hasattr(sample_data, 'edge_attr') and sample_data.edge_attr is not None:
                return sample_data.edge_attr.shape[-1]
            else:
                logger.warning("No edge attributes found in sample data, using 0")
                return 0
                
        except Exception as e:
            logger.warning(f"Could not determine edge dimension: {e}, using 0")
            return 0
    
    def test_model(self, 
                  model: Any, 
                  test_loader: DataLoader, 
                  trainer: Optional[pl.Trainer] = None) -> Dict[str, float]:
        """
        Test the trained model.
        
        Args:
            model: Trained model
            test_loader: Test data loader
            trainer: Optional trainer instance
            
        Returns:
            Dictionary containing test metrics
            
        Raises:
            ValidationError: If testing fails
        """
        logger.info("Starting model testing...")
        self._print_testing_header()
        
        try:
            if trainer is None:
                trainer = self.trainer
            
            if trainer is None:
                raise ValidationError("No trainer available. Train model first or provide trainer.")
            
            # Run testing
            test_results = trainer.test(model, dataloaders=test_loader, verbose=True)
            
            # Extract metrics from results
            if test_results and len(test_results) > 0:
                test_metrics = test_results[0]
                logger.info("Testing completed successfully")
                return test_metrics
            else:
                logger.warning("No test results returned")
                return {}
                
        except Exception as e:
            raise ValidationError(f"Model testing failed: {str(e)}") from e
    
    def predict_with_model(self, 
                          model: Any, 
                          data_loader: DataLoader, 
                          trainer: Optional[pl.Trainer] = None) -> List[torch.Tensor]:
        """
        Make predictions using a trained model.
        
        Args:
            model: Trained model
            data_loader: Data loader for prediction
            trainer: Optional trainer instance
            
        Returns:
            List of prediction tensors
            
        Raises:
            ValidationError: If prediction fails
        """
        logger.info("Starting model prediction...")
        self._print_prediction_header()
        
        try:
            if trainer is None:
                trainer = self.trainer
            
            if trainer is None:
                # Create a lightweight trainer for prediction
                trainer = self._create_prediction_trainer()
            
            # Run prediction
            predictions = trainer.predict(model, dataloaders=data_loader)
            logger.info(f"Prediction completed: {len(predictions)} batches processed")
            
            return predictions
            
        except Exception as e:
            raise ValidationError(f"Model prediction failed: {str(e)}") from e
    
    def _create_prediction_trainer(self) -> pl.Trainer:
        """Create a lightweight trainer for prediction."""
        device_type = self._get_device_type()
        
        return pl.Trainer(
            accelerator=device_type,
            devices=self.training_config.devices,
            precision=self.training_config.precision,
            strategy="auto",
            enable_progress_bar=True,
            enable_model_summary=False,
            logger=False
        )
    
    def run_kfold_training(self, 
                          fold_loaders: List[Tuple[DataLoader, DataLoader]], 
                          test_loader: Optional[DataLoader] = None) -> Dict[str, Any]:
        """
        Run k-fold cross-validation training.
        
        Args:
            fold_loaders: List of (train_loader, val_loader) tuples for each fold
            test_loader: Optional test data loader for final evaluation
            
        Returns:
            Dictionary containing k-fold training results
            
        Raises:
            ModelTrainingError: If k-fold training fails
        """
        logger.info("Starting k-fold cross-validation training...")
        self._print_kfold_header()
        
        try:
            n_folds = len(fold_loaders)
            
            # Store original configuration
            original_exp_name = self.config['experiment']['name']
            original_log_dir = self.config['logging']['log_dir']
            
            # Results storage
            fold_results: List[FoldResult] = []
            fold_models = []
            fold_trainers = []
            
            # Run training for each fold
            for fold_idx, (train_loader, val_loader) in enumerate(fold_loaders):
                fold_result = self._train_single_fold(
                    fold_idx, n_folds, train_loader, val_loader,
                    original_exp_name, original_log_dir
                )
                
                fold_results.append(fold_result)
                
                # Store models and trainers if needed for ensemble
                if self.cv_config.ensemble_prediction:
                    fold_models.append(self.model_manager.model)
                    fold_trainers.append(self.trainer)
            
            # Restore original configuration
            self.config['experiment']['name'] = original_exp_name
            self.config['logging']['log_dir'] = original_log_dir
            
            # Aggregate results
            aggregated_results = self._aggregate_fold_results(fold_results)
            
            # Final test evaluation if test set is available
            test_results = None
            if test_loader is not None and fold_models:
                logger.info("Running final test evaluation...")
                test_results = self._evaluate_on_test_set(fold_models, test_loader)
            
            # Prepare final results
            kfold_results = self._prepare_kfold_results(
                n_folds, fold_results, aggregated_results, test_results, fold_models
            )
            
            # Save results
            if self.cv_config.save_fold_results:
                self._save_kfold_results(kfold_results, original_log_dir)
            
            # Print summary
            self._print_kfold_summary(kfold_results)
            
            logger.info("K-fold cross-validation completed successfully")
            return kfold_results
            
        except Exception as e:
            raise ModelTrainingError(f"K-fold training failed: {str(e)}") from e
    
    def _train_single_fold(self, 
                          fold_idx: int, 
                          n_folds: int,
                          train_loader: DataLoader, 
                          val_loader: DataLoader,
                          original_exp_name: str, 
                          original_log_dir: str) -> FoldResult:
        """Train a single fold and return results."""
        logger.info(f"Training fold {fold_idx + 1}/{n_folds}")
        print(f"\n{TrainingConstants.SEPARATOR_MEDIUM}")
        print(f"TRAINING FOLD {fold_idx + 1}/{n_folds}")
        print(f"{TrainingConstants.SEPARATOR_MEDIUM}")
        
        # Setup fold-specific configuration
        fold_exp_name = f"{original_exp_name}_fold_{fold_idx + 1}"
        fold_log_dir = str(Path(original_log_dir) / f"fold_{fold_idx + 1}")
        
        # Update configuration temporarily
        self.config['experiment']['name'] = fold_exp_name
        self.config['logging']['log_dir'] = fold_log_dir
        
        # Reset state for new fold
        self.model_manager.reset()
        self.trainer = None
        
        # Record timing
        start_time = time.time()
        
        # Train model for this fold
        model, trainer = self.train_model(
            train_loader, val_loader,
            experiment_name=original_exp_name,
            fold_info=f"fold_{fold_idx + 1}"
        )
        
        training_time = time.time() - start_time
        
        # Extract metrics
        val_metrics = self._extract_validation_metrics(trainer)
        
        # Create fold result
        fold_result = FoldResult(
            fold_idx=fold_idx + 1,
            metrics=val_metrics,
            trainer_log_dir=trainer.logger.log_dir if trainer.logger else None,
            training_time=training_time
        )
        
        # Save model checkpoint for this fold
        if self.cv_config.save_fold_results and trainer.logger:
            fold_result.model_path = self._save_fold_checkpoint(
                trainer, fold_idx + 1, trainer.logger.log_dir
            )
        
        logger.info(f"Fold {fold_idx + 1} completed in {training_time:.2f}s. Metrics: {val_metrics}")
        
        return fold_result
    
    def _extract_validation_metrics(self, trainer: pl.Trainer) -> Dict[str, float]:
        """Extract validation metrics from trainer."""
        val_metrics = {}
        
        if trainer.callback_metrics:
            for key, value in trainer.callback_metrics.items():
                if isinstance(value, torch.Tensor):
                    val_metrics[key] = value.item()
                else:
                    val_metrics[key] = float(value) if isinstance(value, (int, float)) else value
        
        return val_metrics
    
    def _save_fold_checkpoint(self, 
                             trainer: pl.Trainer, 
                             fold_idx: int, 
                             log_dir: str) -> str:
        """Save checkpoint for a specific fold."""
        try:
            checkpoint_dir = Path(log_dir) / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            model_path = checkpoint_dir / TrainingConstants.FOLD_CHECKPOINT_TEMPLATE.format(fold_idx)
            trainer.save_checkpoint(str(model_path))
            
            logger.debug(f"Fold {fold_idx} checkpoint saved to: {model_path}")
            return str(model_path)
            
        except Exception as e:
            logger.warning(f"Failed to save fold {fold_idx} checkpoint: {e}")
            return None
    
    def _aggregate_fold_results(self, fold_results: List[FoldResult]) -> Dict[str, MetricStatistics]:
        """Aggregate metrics across all folds."""
        if not fold_results:
            return {}
        
        # Collect all metric names
        all_metrics = set()
        for fold_result in fold_results:
            all_metrics.update(fold_result.metrics.keys())
        
        aggregated = {}
        for metric_name in all_metrics:
            values = []
            for fold_result in fold_results:
                if metric_name in fold_result.metrics:
                    values.append(fold_result.metrics[metric_name])
            
            if values:
                aggregated[metric_name] = MetricStatistics(
                    mean=float(np.mean(values)),
                    std=float(np.std(values)),
                    min=float(np.min(values)),
                    max=float(np.max(values)),
                    values=values
                )
        
        return aggregated
    
    def _evaluate_on_test_set(self, fold_models: List[Any], test_loader: DataLoader) -> Dict[str, Any]:
        """Evaluate all fold models on the test set."""
        logger.info("Evaluating ensemble on test set...")
        
        device_type = self._get_device_type()
        device = torch.device(device_type)
        
        all_predictions = []
        true_labels = []
        
        # Collect predictions from all models
        for batch in test_loader:
            batch = batch.to(device)
            batch_predictions = []
            
            # Get predictions from each fold model
            for model in fold_models:
                model.eval()
                model.to(device)
                with torch.no_grad():
                    pred = model(batch)
                    batch_predictions.append(pred.cpu())
            
            # Store predictions and labels
            all_predictions.append(torch.stack(batch_predictions))
            true_labels.append(batch.y.cpu())
        
        # Process predictions and calculate metrics
        return self._process_test_predictions(all_predictions, true_labels, fold_models)
    
    def _process_test_predictions(self, 
                                all_predictions: List[torch.Tensor],
                                true_labels: List[torch.Tensor],
                                fold_models: List[Any]) -> Dict[str, Any]:
        """Process test predictions and calculate metrics."""
        # Concatenate all batches
        all_predictions = torch.cat(all_predictions, dim=1)  # [n_folds, total_samples, n_tasks]
        true_labels = torch.cat(true_labels, dim=0)  # [total_samples, n_tasks]
        
        # Calculate ensemble predictions (mean across folds)
        ensemble_pred = torch.mean(all_predictions, dim=0)  # [total_samples, n_tasks]
        
        # Calculate metrics based on task type
        task_type = TaskType(self.config['model']['task'])
        
        if task_type == TaskType.REGRESSION:
            ensemble_metrics = MetricsCalculator.calculate_regression_metrics(ensemble_pred, true_labels)
        else:  # Classification
            ensemble_metrics = MetricsCalculator.calculate_classification_metrics(ensemble_pred, true_labels)
        
        # Calculate individual model metrics
        individual_metrics = []
        for fold_idx in range(len(fold_models)):
            fold_pred = all_predictions[fold_idx]
            
            if task_type == TaskType.REGRESSION:
                fold_metrics = MetricsCalculator.calculate_regression_metrics(fold_pred, true_labels)
            else:
                fold_metrics = MetricsCalculator.calculate_classification_metrics(fold_pred, true_labels)
            
            fold_metrics['fold'] = fold_idx + 1
            individual_metrics.append(fold_metrics)
        
        return {
            'ensemble_predictions': ensemble_pred.numpy(),
            'individual_predictions': all_predictions.numpy(),
            'true_labels': true_labels.numpy(),
            'ensemble_metrics': ensemble_metrics,
            'individual_metrics': individual_metrics
        }
    
    def _prepare_kfold_results(self,
                              n_folds: int,
                              fold_results: List[FoldResult],
                              aggregated_results: Dict[str, MetricStatistics],
                              test_results: Optional[Dict[str, Any]],
                              fold_models: List[Any]) -> Dict[str, Any]:
        """Prepare final k-fold results dictionary."""
        return {
            'n_folds': n_folds,
            'fold_results': [
                {
                    'fold': result.fold_idx,
                    'metrics': result.metrics,
                    'model_path': result.model_path,
                    'trainer_log_dir': result.trainer_log_dir,
                    'training_time': result.training_time,
                    'total_time': result.total_time
                }
                for result in fold_results
            ],
            'aggregated_metrics': {
                metric_name: {
                    'mean': stats.mean,
                    'std': stats.std,
                    'min': stats.min,
                    'max': stats.max,
                    'values': stats.values,
                    'confidence_interval': stats.confidence_interval()
                }
                for metric_name, stats in aggregated_results.items()
            },
            'test_results': test_results,
            'best_fold': self._find_best_fold(fold_results),
            'fold_models': fold_models if not self.cv_config.save_fold_results else None,
            'config': self.config.copy(),
            'total_training_time': sum(r.training_time for r in fold_results if r.training_time)
        }
    
    def _save_kfold_results(self, kfold_results: Dict[str, Any], log_dir: str) -> None:
        """Save k-fold results to file."""
        try:
            results_path = Path(log_dir) / TrainingConstants.KFOLD_RESULTS_FILE
            results_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(results_path, 'wb') as f:
                pickle.dump(kfold_results, f)
            
            logger.info(f"K-fold results saved to: {results_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save k-fold results: {e}")
    
    def _find_best_fold(self, fold_results: List[FoldResult]) -> Optional[Dict[str, Any]]:
        """Find the best performing fold based on the monitoring metric."""
        if not fold_results:
            return None
        
        monitor_metric = self.logging_config.monitor
        monitor_mode = MetricMode(self.logging_config.mode)
        
        best_fold = None
        best_value = None
        
        for fold_result in fold_results:
            if monitor_metric in fold_result.metrics:
                value = fold_result.metrics[monitor_metric]
                
                if best_value is None:
                    best_value = value
                    best_fold = fold_result
                elif ((monitor_mode == MetricMode.MINIMIZE and value < best_value) or
                      (monitor_mode == MetricMode.MAXIMIZE and value > best_value)):
                    best_value = value
                    best_fold = fold_result
        
        if best_fold:
            return {
                'fold': best_fold.fold_idx,
                'metrics': best_fold.metrics,
                'model_path': best_fold.model_path,
                'trainer_log_dir': best_fold.trainer_log_dir,
                'training_time': best_fold.training_time
            }
        
        return None
    
    def _print_training_header(self) -> None:
        """Print training header."""
        print(TrainingConstants.SEPARATOR_MEDIUM)
        print("TRAINING MODEL")
        print(TrainingConstants.SEPARATOR_MEDIUM)
    
    def _print_testing_header(self) -> None:
        """Print testing header."""
        print(TrainingConstants.SEPARATOR_MEDIUM)
        print("TESTING MODEL")
        print(TrainingConstants.SEPARATOR_MEDIUM)
    
    def _print_prediction_header(self) -> None:
        """Print prediction header."""
        print(TrainingConstants.SEPARATOR_MEDIUM)
        print("MAKING PREDICTIONS")
        print(TrainingConstants.SEPARATOR_MEDIUM)
    
    def _print_kfold_header(self) -> None:
        """Print k-fold header."""
        print(TrainingConstants.SEPARATOR_LONG)
        print("RUNNING K-FOLD CROSS-VALIDATION TRAINING")
        print(TrainingConstants.SEPARATOR_LONG)
    
    def _print_kfold_summary(self, kfold_results: Dict[str, Any]) -> None:
        """Print a comprehensive summary of k-fold cross-validation results."""
        print(f"\n{TrainingConstants.SEPARATOR_LONG}")
        print("K-FOLD CROSS-VALIDATION SUMMARY")
        print(f"{TrainingConstants.SEPARATOR_LONG}")
        
        n_folds = kfold_results['n_folds']
        total_time = kfold_results.get('total_training_time', 0)
        
        print(f"Number of folds: {n_folds}")
        print(f"Total training time: {total_time:.2f} seconds")
        print(f"Average time per fold: {total_time/n_folds:.2f} seconds")
        
        # Print aggregated metrics
        aggregated = kfold_results['aggregated_metrics']
        if aggregated:
            print(f"\nCross-validation metrics (mean ± std):")
            for metric_name, stats in aggregated.items():
                ci_lower, ci_upper = stats['confidence_interval']
                print(f"  {metric_name}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                      f"(min: {stats['min']:.4f}, max: {stats['max']:.4f}) "
                      f"[95% CI: {ci_lower:.4f}, {ci_upper:.4f}]")
        
        # Print best fold information
        best_fold = kfold_results['best_fold']
        if best_fold:
            print(f"\nBest fold: {best_fold['fold']}")
            print(f"Best fold metrics:")
            for metric_name, value in best_fold['metrics'].items():
                print(f"  {metric_name}: {value:.4f}")
            
            if best_fold.get('training_time'):
                print(f"  training_time: {best_fold['training_time']:.2f}s")
        
        # Print test results if available
        test_results = kfold_results['test_results']
        if test_results:
            print(f"\nTest set evaluation (ensemble):")
            for metric_name, value in test_results['ensemble_metrics'].items():
                print(f"  {metric_name}: {value:.4f}")
            
            print(f"\nTest set evaluation (individual folds):")
            for fold_metrics in test_results['individual_metrics']:
                fold_num = fold_metrics['fold']
                metrics_str = ", ".join([
                    f"{k}: {v:.4f}" for k, v in fold_metrics.items() 
                    if k != 'fold'
                ])
                print(f"  Fold {fold_num}: {metrics_str}")
        
        print(f"\n{TrainingConstants.SEPARATOR_LONG}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive training summary.
        
        Returns:
            Dictionary containing training configuration and status
        """
        summary = {
            'training_config': {
                'epochs': self.training_config.epochs,
                'devices': self.training_config.devices,
                'precision': self.training_config.precision,
                'accelerator': self.training_config.accelerator,
                'strategy': self.training_config.strategy
            },
            'logging_config': {
                'log_dir': self.logging_config.log_dir,
                'monitor': self.logging_config.monitor,
                'mode': self.logging_config.mode,
                'save_top_k': self.logging_config.save_top_k
            },
            'early_stopping': {
                'enabled': self.early_stopping_config.enabled,
                'patience': self.early_stopping_config.patience,
                'monitor': self.early_stopping_config.monitor
            },
            'cross_validation': {
                'enabled': self.cv_config.enabled,
                'k_folds': self.cv_config.k_folds,
                'save_fold_results': self.cv_config.save_fold_results
            },
            'trainer_available': self.trainer is not None,
            'device_type': self._get_device_type()
        }
        
        return summary
    
    def reset(self) -> None:
        """Reset the training manager state."""
        logger.info("Resetting TrainingManager state...")
        self.trainer = None
        logger.info("TrainingManager state reset successfully") 