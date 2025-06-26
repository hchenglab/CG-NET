#!/usr/bin/env python
"""Model management for CG-NET training pipeline.

This module provides comprehensive model management capabilities including:
- Model creation and initialization
- Checkpoint loading and saving
- Model state management
- Checkpoint discovery and selection
- Model architecture information and summaries
- Configuration validation and error handling

The ModelManager class serves as the central hub for all model-related operations
in the CG-NET training pipeline.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
import logging

import torch
import pytorch_lightning as pl

from ..models import CGNET

# Configure logging
logger = logging.getLogger(__name__)


class CheckpointType(Enum):
    """Enumeration of checkpoint types."""
    LATEST = "latest"
    BEST = "best"
    CUSTOM = "custom"


class ModelConstants:
    """Constants used throughout model management."""
    
    # Default values
    DEFAULT_LR = 0.001
    DEFAULT_EPOCHS = 300
    DEFAULT_LOG_DIR = "logs"
    DEFAULT_CHECKPOINTS_DIR = "checkpoints"
    
    # File patterns
    CHECKPOINT_PATTERN = "**/*.ckpt"
    CHECKPOINT_DIR_PATTERN = "**/checkpoints/*.ckpt"
    
    # File extensions
    CHECKPOINT_EXTENSION = ".ckpt"
    MODEL_STATE_EXTENSION = ".pth"
    
    # Metric modes
    MINIMIZE_MODE = "min"
    MAXIMIZE_MODE = "max"
    
    # Default monitoring
    DEFAULT_MONITOR_METRIC = "val_loss"
    DEFAULT_MONITOR_MODE = "min"


class ModelError(Exception):
    """Base exception for model-related errors."""
    pass


class ModelConfigurationError(ModelError):
    """Exception raised for model configuration errors."""
    pass


class CheckpointError(ModelError):
    """Exception raised for checkpoint-related errors."""
    pass


class ModelValidator:
    """Handles model configuration validation."""
    
    @staticmethod
    def validate_model_config(config: Dict[str, Any]) -> None:
        """
        Validate model configuration.
        
        Args:
            config: Configuration dictionary to validate
            
        Raises:
            ModelConfigurationError: If configuration is invalid
        """
        if 'model' not in config:
            raise ModelConfigurationError("Missing 'model' section in configuration")
        
        model_config = config['model']
        required_keys = [
            'in_node_dim', 'hidden_node_dim', 'predictor_hidden_dim',
            'num_conv', 'n_h', 'n_tasks', 'task', 'n_classes'
        ]
        
        missing_keys = [key for key in required_keys if key not in model_config]
        if missing_keys:
            raise ModelConfigurationError(f"Missing required model configuration keys: {missing_keys}")
        
        # Validate task type
        task = model_config.get('task')
        if task not in ['regression', 'classification']:
            raise ModelConfigurationError(f"Invalid task type: {task}. Must be 'regression' or 'classification'")
        
        # Validate classification-specific requirements
        if task == 'classification':
            n_classes = model_config.get('n_classes', 0)
            if n_classes <= 1:
                raise ModelConfigurationError(f"For classification tasks, n_classes must be > 1, got {n_classes}")
        
        # Validate positive values
        positive_keys = ['in_node_dim', 'hidden_node_dim', 'predictor_hidden_dim', 'num_conv', 'n_h', 'n_tasks']
        for key in positive_keys:
            value = model_config.get(key, 0)
            if value <= 0:
                raise ModelConfigurationError(f"Model parameter '{key}' must be positive, got {value}")


class CheckpointManager:
    """Handles checkpoint discovery and selection."""
    
    @staticmethod
    def find_latest_checkpoint(log_dir: str) -> Optional[str]:
        """
        Find the latest checkpoint by modification time.
        
        Args:
            log_dir: Directory to search for checkpoints
            
        Returns:
            Path to the latest checkpoint, or None if not found
        """
        logs_path = Path(log_dir)
        if not logs_path.exists():
            logger.warning(f"Log directory does not exist: {log_dir}")
            return None
        
        checkpoints = list(logs_path.glob(ModelConstants.CHECKPOINT_PATTERN))
        
        if not checkpoints:
            logger.info(f"No checkpoints found in {log_dir}")
            return None
        
        # Sort by modification time and get the latest
        latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
        logger.info(f"Found latest checkpoint: {latest_checkpoint}")
        
        return str(latest_checkpoint)
    
    @staticmethod
    def find_best_checkpoint(log_dir: str, 
                           experiment_name: Optional[str] = None,
                           monitor_metric: str = ModelConstants.DEFAULT_MONITOR_METRIC,
                           mode: str = ModelConstants.DEFAULT_MONITOR_MODE) -> Optional[str]:
        """
        Find the best checkpoint based on monitoring metric.
        
        Args:
            log_dir: Directory to search for checkpoints
            experiment_name: Specific experiment name to search in
            monitor_metric: Metric to use for selecting best checkpoint
            mode: Either 'min' or 'max' for metric optimization
            
        Returns:
            Path to the best checkpoint, or None if not found
        """
        logs_path = Path(log_dir)
        search_dir = logs_path / experiment_name if experiment_name else logs_path
        
        if not search_dir.exists():
            logger.warning(f"Search directory does not exist: {search_dir}")
            return None
        
        checkpoints = list(search_dir.glob(ModelConstants.CHECKPOINT_DIR_PATTERN))
        
        if not checkpoints:
            logger.info(f"No checkpoints found in {search_dir}")
            return None
        
        # Try to find the best checkpoint based on filename patterns
        best_checkpoint = CheckpointManager._select_best_by_metric(
            checkpoints, monitor_metric, mode
        )
        
        if best_checkpoint:
            logger.info(f"Found best checkpoint: {best_checkpoint}")
            return str(best_checkpoint)
        else:
            # Fallback to the first checkpoint found
            fallback = str(checkpoints[0])
            logger.warning(f"Could not determine best checkpoint, using: {fallback}")
            return fallback
    
    @staticmethod
    def _select_best_by_metric(checkpoints: List[Path], 
                             monitor_metric: str, 
                             mode: str) -> Optional[Path]:
        """Select the best checkpoint based on metric value in filename."""
        best_checkpoint = None
        best_value = None
        
        for checkpoint in checkpoints:
            value = CheckpointManager._extract_metric_from_filename(
                checkpoint.stem, monitor_metric
            )
            
            if value is not None:
                if best_value is None:
                    best_value = value
                    best_checkpoint = checkpoint
                elif CheckpointManager._is_better_value(value, best_value, mode):
                    best_value = value
                    best_checkpoint = checkpoint
        
        return best_checkpoint
    
    @staticmethod
    def _extract_metric_from_filename(filename: str, monitor_metric: str) -> Optional[float]:
        """Extract metric value from checkpoint filename."""
        try:
            # Handle different separators in metric names
            metric_variants = [monitor_metric, monitor_metric.replace('_', '-')]
            
            parts = filename.split('-')
            for part in parts:
                for variant in metric_variants:
                    if variant in part:
                        value_str = part.split('=')[-1]
                        return float(value_str)
        except (ValueError, IndexError):
            pass
        
        return None
    
    @staticmethod
    def _is_better_value(value: float, best_value: float, mode: str) -> bool:
        """Check if a value is better than the current best based on mode."""
        if mode == ModelConstants.MINIMIZE_MODE:
            return value < best_value
        elif mode == ModelConstants.MAXIMIZE_MODE:
            return value > best_value
        else:
            logger.warning(f"Unknown mode: {mode}, defaulting to minimize")
            return value < best_value


class ModelManager:
    """Manages model creation, loading, and checkpointing for CG-NET training.
    
    This class provides a comprehensive interface for model management including:
    - Model creation and initialization with configuration validation
    - Checkpoint loading and saving with error handling
    - Model state management and persistence
    - Checkpoint discovery and selection strategies
    - Model architecture information and summaries
    
    Attributes:
        config (Dict[str, Any]): Configuration dictionary
        model (Optional[CGNET]): Current model instance
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ModelManager with configuration.
        
        Args:
            config: Configuration dictionary containing model and training settings
            
        Raises:
            ModelConfigurationError: If configuration is invalid
        """
        self.config = config
        self.model: Optional[CGNET] = None
        
        # Validate configuration
        ModelValidator.validate_model_config(config)
        
        logger.info("ModelManager initialized successfully")
    
    def create_model(self, in_edge_dim: int) -> CGNET:
        """
        Create CG-NET model based on configuration.
        
        Args:
            in_edge_dim: Input edge feature dimension
            
        Returns:
            Created CGNET model instance
            
        Raises:
            ModelConfigurationError: If model creation fails
        """
        logger.info(f"Creating CG-NET model with edge dimension: {in_edge_dim}")
        
        try:
            # Update edge dimension in config
            self.config['model']['in_edge_dim'] = in_edge_dim
            
            model_config = self.config['model']
            training_params = self._get_training_parameters()
            
            model = CGNET(
                in_node_dim=model_config['in_node_dim'],
                hidden_node_dim=model_config['hidden_node_dim'],
                in_edge_dim=in_edge_dim,
                predictor_hidden_dim=model_config['predictor_hidden_dim'],
                n_conv=model_config['num_conv'],
                n_h=model_config['n_h'],
                n_tasks=model_config['n_tasks'],
                task=model_config['task'],
                n_classes=model_config['n_classes'],
                lr=training_params['lr'],
                tmax=training_params['tmax']
            )
            
            self.model = model
            logger.info("CG-NET model created successfully")
            logger.debug(f"Model parameters: {self._count_parameters(model)}")
            
            return model
            
        except Exception as e:
            raise ModelConfigurationError(f"Failed to create model: {str(e)}") from e
    
    def _get_training_parameters(self) -> Dict[str, Any]:
        """Get training parameters with fallback to defaults."""
        training_config = self.config.get('training', {})
        
        lr = training_config.get('lr', ModelConstants.DEFAULT_LR)
        epochs = training_config.get('epochs', ModelConstants.DEFAULT_EPOCHS)
        tmax = training_config.get('tmax', epochs)
        
        return {'lr': lr, 'tmax': tmax}
    
    def _count_parameters(self, model: CGNET) -> Dict[str, int]:
        """Count model parameters."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'non_trainable': total_params - trainable_params
        }
    
    def load_model_from_checkpoint(self, checkpoint_path: str) -> CGNET:
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint
            
        Returns:
            Loaded CGNET model instance
            
        Raises:
            CheckpointError: If checkpoint loading fails
            FileNotFoundError: If checkpoint file doesn't exist
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        
        try:
            model_config = self.config['model']
            training_params = self._get_training_parameters()
            
            model = CGNET.load_from_checkpoint(
                str(checkpoint_path),
                in_node_dim=model_config['in_node_dim'],
                hidden_node_dim=model_config['hidden_node_dim'],
                in_edge_dim=model_config['in_edge_dim'],
                predictor_hidden_dim=model_config['predictor_hidden_dim'],
                n_conv=model_config['num_conv'],
                n_h=model_config['n_h'],
                n_tasks=model_config['n_tasks'],
                task=model_config['task'],
                n_classes=model_config['n_classes'],
                lr=training_params['lr'],
                tmax=training_params['tmax']
            )
            
            self.model = model
            logger.info("Model loaded from checkpoint successfully")
            
            return model
            
        except Exception as e:
            raise CheckpointError(f"Failed to load model from checkpoint {checkpoint_path}: {str(e)}") from e
    
    def get_model_summary(self) -> str:
        """
        Get a comprehensive summary of the model architecture.
        
        Returns:
            Formatted string containing model information
        """
        if self.model is None:
            return "No model loaded"
        
        model_config = self.config['model']
        
        # Build summary sections
        summary_lines = [
            "CG-NET Model Configuration:",
            "=" * 50,
            f"Task Type: {model_config['task']}",
            f"Input node features: {model_config['in_node_dim']}",
            f"Hidden node features: {model_config['hidden_node_dim']}",
            f"Input edge features: {model_config.get('in_edge_dim', 'Not set')}",
            f"Predictor hidden features: {model_config['predictor_hidden_dim']}",
            f"Number of conv layers: {model_config['num_conv']}",
            f"Number of heads: {model_config['n_h']}",
            f"Number of tasks: {model_config['n_tasks']}"
        ]
        
        # Add classification-specific info
        if model_config['task'] == 'classification':
            summary_lines.append(f"Number of classes: {model_config['n_classes']}")
        
        # Add parameter counts
        if self.model is not None:
            param_counts = self._count_parameters(self.model)
            summary_lines.extend([
                "",
                "Parameter Statistics:",
                "-" * 30,
                f"Total parameters: {param_counts['total']:,}",
                f"Trainable parameters: {param_counts['trainable']:,}",
                f"Non-trainable parameters: {param_counts['non_trainable']:,}"
            ])
        
        return "\n".join(summary_lines)
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """
        Find the latest model checkpoint.
        
        Returns:
            Path to the latest checkpoint, or None if no checkpoints found
        """
        log_dir = self._get_log_directory()
        return CheckpointManager.find_latest_checkpoint(log_dir)
    
    def get_best_checkpoint(self, experiment_name: Optional[str] = None) -> Optional[str]:
        """
        Find the best model checkpoint based on the monitoring metric.
        
        Args:
            experiment_name: Specific experiment name to search in
            
        Returns:
            Path to the best checkpoint, or None if not found
        """
        log_dir = self._get_log_directory()
        logging_config = self.config.get('logging', {})
        
        monitor_metric = logging_config.get('monitor', ModelConstants.DEFAULT_MONITOR_METRIC)
        mode = logging_config.get('mode', ModelConstants.DEFAULT_MONITOR_MODE)
        
        return CheckpointManager.find_best_checkpoint(
            log_dir, experiment_name, monitor_metric, mode
        )
    
    def _get_log_directory(self) -> str:
        """Get log directory from configuration."""
        logging_config = self.config.get('logging', {})
        return logging_config.get('log_dir', ModelConstants.DEFAULT_LOG_DIR)
    
    def save_model_state(self, save_path: str, include_config: bool = True) -> None:
        """
        Save current model state to file.
        
        Args:
            save_path: Path to save the model
            include_config: Whether to include configuration in the saved file
            
        Raises:
            ModelError: If no model is available or saving fails
        """
        if self.model is None:
            raise ModelError("No model to save")
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving model state to: {save_path}")
        
        try:
            # Prepare save data
            save_data = {
                'model_state_dict': self.model.state_dict(),
                'model_config': self.config['model']
            }
            
            if include_config:
                save_data['config'] = self.config
            
            # Save to file
            torch.save(save_data, save_path)
            logger.info("Model state saved successfully")
            
        except Exception as e:
            raise ModelError(f"Failed to save model to {save_path}: {str(e)}") from e
    
    def load_model_state(self, load_path: str) -> CGNET:
        """
        Load model state from saved file.
        
        Args:
            load_path: Path to load the model from
            
        Returns:
            Loaded CGNET model instance
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            ModelError: If loading fails
        """
        load_path = Path(load_path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        logger.info(f"Loading model state from: {load_path}")
        
        try:
            checkpoint = torch.load(load_path, map_location='cpu')
            
            # Update config if saved in checkpoint
            if 'config' in checkpoint:
                self.config.update(checkpoint['config'])
                logger.debug("Configuration updated from saved file")
            
            # Get model configuration
            model_config = checkpoint.get('model_config', self.config['model'])
            training_params = self._get_training_parameters()
            
            # Create model
            model = CGNET(
                in_node_dim=model_config['in_node_dim'],
                hidden_node_dim=model_config['hidden_node_dim'],
                in_edge_dim=model_config['in_edge_dim'],
                predictor_hidden_dim=model_config['predictor_hidden_dim'],
                n_conv=model_config['num_conv'],
                n_h=model_config['n_h'],
                n_tasks=model_config['n_tasks'],
                task=model_config['task'],
                n_classes=model_config['n_classes'],
                lr=training_params['lr'],
                tmax=training_params['tmax']
            )
            
            # Load state dict
            model.load_state_dict(checkpoint['model_state_dict'])
            
            self.model = model
            logger.info("Model state loaded successfully")
            
            return model
            
        except Exception as e:
            raise ModelError(f"Failed to load model from {load_path}: {str(e)}") from e
    
    def get_checkpoint_info(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Get information about a checkpoint file.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            
        Returns:
            Dictionary containing checkpoint information
            
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            CheckpointError: If checkpoint cannot be read
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        try:
            # Get file statistics
            stat = checkpoint_path.stat()
            
            info = {
                'path': str(checkpoint_path),
                'size_mb': stat.st_size / (1024 * 1024),
                'modified_time': stat.st_mtime,
                'exists': True
            }
            
            # Try to load and get additional info
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                
                if hasattr(checkpoint, 'keys'):
                    info['checkpoint_keys'] = list(checkpoint.keys())
                
                if 'epoch' in checkpoint:
                    info['epoch'] = checkpoint['epoch']
                
                if 'global_step' in checkpoint:
                    info['global_step'] = checkpoint['global_step']
                
            except Exception as e:
                logger.warning(f"Could not read checkpoint details: {e}")
                info['read_error'] = str(e)
            
            return info
            
        except Exception as e:
            raise CheckpointError(f"Failed to get checkpoint info for {checkpoint_path}: {str(e)}") from e
    
    def list_available_checkpoints(self, experiment_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all available checkpoints with their information.
        
        Args:
            experiment_name: Specific experiment name to search in
            
        Returns:
            List of dictionaries containing checkpoint information
        """
        log_dir = self._get_log_directory()
        logs_path = Path(log_dir)
        search_dir = logs_path / experiment_name if experiment_name else logs_path
        
        if not search_dir.exists():
            logger.warning(f"Search directory does not exist: {search_dir}")
            return []
        
        checkpoints = list(search_dir.glob(ModelConstants.CHECKPOINT_PATTERN))
        checkpoint_info = []
        
        for checkpoint in checkpoints:
            try:
                info = self.get_checkpoint_info(str(checkpoint))
                checkpoint_info.append(info)
            except Exception as e:
                logger.warning(f"Could not get info for checkpoint {checkpoint}: {e}")
        
        # Sort by modification time (newest first)
        checkpoint_info.sort(key=lambda x: x.get('modified_time', 0), reverse=True)
        
        return checkpoint_info
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the current model.
        
        Returns:
            Dictionary containing model information
        """
        info = {
            'model_loaded': self.model is not None,
            'config': self.config.get('model', {}),
            'task_type': self.config.get('model', {}).get('task', 'unknown')
        }
        
        if self.model is not None:
            param_counts = self._count_parameters(self.model)
            info.update({
                'parameters': param_counts,
                'model_type': type(self.model).__name__,
                'device': str(next(self.model.parameters()).device) if list(self.model.parameters()) else 'unknown'
            })
        
        return info
    
    def reset(self) -> None:
        """Reset the model manager state."""
        logger.info("Resetting ModelManager state...")
        self.model = None
        logger.info("ModelManager state reset successfully") 