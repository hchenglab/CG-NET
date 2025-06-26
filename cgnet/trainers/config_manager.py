#!/usr/bin/env python
"""Unified configuration management for CG-NET training pipeline.

This module provides comprehensive configuration management capabilities including:
- Configuration loading, validation, and saving
- Template generation for different tasks
- CLI parameter override handling
- Configuration optimization suggestions
- Error checking and validation

The ConfigManager class serves as the central hub for all configuration-related
operations in the CG-NET training pipeline.
"""

import os
import yaml
import copy
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from enum import Enum
import logging

# Configure logging
logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Enumeration of supported task types."""
    REGRESSION = "regression"
    CLASSIFICATION = "classification"


class TemplateType(Enum):
    """Enumeration of available configuration templates."""
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    PREDICTION = "prediction"


class FeaturizerMethod(Enum):
    """Enumeration of supported featurizer methods."""
    CR = "CR"
    NTH_NN = "nth-NN"


class ValidationSeverity(Enum):
    """Enumeration of validation message severity levels."""
    WARNING = "warning"
    ERROR = "error"


class ConfigConstants:
    """Constants used throughout configuration management."""
    
    # Default values
    DEFAULT_CONFIG_NAME = "config.yml"
    DEFAULT_TRAIN_RATIO = 0.8
    DEFAULT_VAL_RATIO = 0.1
    DEFAULT_TEST_RATIO = 0.1
    DEFAULT_SEED = 42
    DEFAULT_EPOCHS = 300
    DEFAULT_BATCH_SIZE = 64
    DEFAULT_LEARNING_RATE = 0.001
    DEFAULT_FOLDS = 5
    
    # Validation thresholds
    MIN_RATIO_SUM_TOLERANCE = 1e-6
    HIGH_EPOCHS_THRESHOLD = 1000
    LOW_EPOCHS_THRESHOLD = 10
    HIGH_LR_THRESHOLD = 0.1
    LOW_LR_THRESHOLD = 1e-6
    LARGE_BATCH_THRESHOLD = 512
    SMALL_BATCH_THRESHOLD = 8
    MIN_FOLDS = 3
    MAX_FOLDS = 10
    
    # Required sections
    BASIC_REQUIRED_SECTIONS = ['experiment', 'data', 'model', 'featurizer']
    TRAINING_REQUIRED_SECTIONS = BASIC_REQUIRED_SECTIONS + ['training', 'logging']
    PREDICTION_REQUIRED_SECTIONS = BASIC_REQUIRED_SECTIONS + ['prediction']
    
    # Memory units
    MEMORY_UNITS = ['GB', 'MB', 'TB']


class ValidationMessage:
    """Represents a validation message with severity and context."""
    
    def __init__(self, severity: ValidationSeverity, message: str, section: str = None):
        self.severity = severity
        self.message = message
        self.section = section
    
    def __str__(self) -> str:
        if self.section:
            return f"[{self.section}] {self.message}"
        return self.message


class ConfigValidator:
    """Handles all configuration validation logic."""
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """
        Comprehensive configuration validation.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            Tuple of (warnings, errors) - Lists of warning and error messages
        """
        warnings_list = []
        errors = []
        
        # Validate basic structure
        structure_warnings, structure_errors = ConfigValidator._validate_structure(config)
        warnings_list.extend(structure_warnings)
        errors.extend(structure_errors)
        
        # Only continue validation if structure is valid
        if not structure_errors:
            # Validate task-specific settings
            task_warnings = ConfigValidator._validate_task_settings(config)
            warnings_list.extend(task_warnings)
            
            # Validate training parameters
            training_warnings = ConfigValidator._validate_training_params(config)
            warnings_list.extend(training_warnings)
            
            # Validate data parameters
            data_warnings, data_errors = ConfigValidator._validate_data_params(config)
            warnings_list.extend(data_warnings)
            errors.extend(data_errors)
            
            # Validate cross-validation settings
            cv_warnings = ConfigValidator._validate_cv_settings(config)
            warnings_list.extend(cv_warnings)
            
            # Validate SLURM settings
            slurm_warnings = ConfigValidator._validate_slurm_settings(config)
            warnings_list.extend(slurm_warnings)
        
        return warnings_list, errors
    
    @staticmethod
    def _validate_structure(config: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Validate basic configuration structure."""
        warnings_list = []
        errors = []
        
        # Determine if this is a prediction-only config
        is_prediction_config = ConfigValidator._is_prediction_only_config(config)
        
        # Select required sections based on config type
        if is_prediction_config:
            required_sections = ConfigConstants.PREDICTION_REQUIRED_SECTIONS
        else:
            required_sections = ConfigConstants.TRAINING_REQUIRED_SECTIONS
        
        # Check required sections
        for section in required_sections:
            if section not in config:
                errors.append(f"Missing required configuration section: '{section}'")
        
        # Validate individual sections
        errors.extend(ConfigValidator._validate_experiment_section(config))
        errors.extend(ConfigValidator._validate_model_section(config))
        errors.extend(ConfigValidator._validate_training_section(config, is_prediction_config))
        errors.extend(ConfigValidator._validate_data_section(config, is_prediction_config))
        errors.extend(ConfigValidator._validate_featurizer_section(config))
        errors.extend(ConfigValidator._validate_logging_section(config, is_prediction_config))
        
        return warnings_list, errors
    
    @staticmethod
    def _is_prediction_only_config(config: Dict[str, Any]) -> bool:
        """Check if this is a prediction-only configuration."""
        # Check if this has prediction section and minimal training configuration
        has_prediction = 'prediction' in config
        has_minimal_training = (
            'training' in config and 
            config.get('training', {}).get('epochs', 0) <= 1
        )
        return has_prediction and has_minimal_training
    
    @staticmethod
    def _validate_experiment_section(config: Dict[str, Any]) -> List[str]:
        """Validate experiment section."""
        errors = []
        
        if 'experiment' not in config:
            return errors
        
        exp_config = config['experiment']
        
        if 'name' not in exp_config or not exp_config['name']:
            errors.append("experiment.name is required and cannot be empty")
        
        if 'seed' not in exp_config:
            errors.append("experiment.seed is required")
        
        return errors
    
    @staticmethod
    def _validate_model_section(config: Dict[str, Any]) -> List[str]:
        """Validate model section."""
        errors = []
        
        if 'model' not in config:
            return errors
        
        model_config = config['model']
        required_keys = ['task', 'n_tasks']
        
        for key in required_keys:
            if key not in model_config:
                errors.append(f"model.{key} is required")
        
        # Validate task type
        task = model_config.get('task')
        if task and task not in [TaskType.REGRESSION.value, TaskType.CLASSIFICATION.value]:
            errors.append(f"model.task must be '{TaskType.REGRESSION.value}' or '{TaskType.CLASSIFICATION.value}'")
        
        # Validate classification-specific requirements
        if task == TaskType.CLASSIFICATION.value and 'n_classes' not in model_config:
            errors.append("model.n_classes is required for classification tasks")
        
        return errors
    
    @staticmethod
    def _validate_training_section(config: Dict[str, Any], is_prediction_config: bool) -> List[str]:
        """Validate training section."""
        errors = []
        
        if is_prediction_config or 'training' not in config:
            return errors
        
        training_config = config['training']
        required_keys = ['epochs', 'batch_size', 'lr']
        
        for key in required_keys:
            if key not in training_config:
                errors.append(f"training.{key} is required")
        
        return errors
    
    @staticmethod
    def _validate_data_section(config: Dict[str, Any], is_prediction_config: bool) -> List[str]:
        """Validate data section."""
        errors = []
        
        if 'data' not in config:
            return errors
        
        data_config = config['data']
        
        # For prediction configs, only path is required
        if is_prediction_config:
            required_keys = ['path']
        else:
            required_keys = ['path', 'train_ratio', 'val_ratio', 'test_ratio']
        
        for key in required_keys:
            if key not in data_config:
                errors.append(f"data.{key} is required")
        
        return errors
    
    @staticmethod
    def _validate_featurizer_section(config: Dict[str, Any]) -> List[str]:
        """Validate featurizer section."""
        errors = []
        
        if 'featurizer' not in config:
            return errors
        
        featurizer_config = config['featurizer']
        required_keys = ['method', 'neighbor_radius', 'max_neighbors']
        
        for key in required_keys:
            if key not in featurizer_config:
                errors.append(f"featurizer.{key} is required")
        
        # Validate method-specific requirements
        method = featurizer_config.get('method')
        if method == FeaturizerMethod.CR.value and 'cluster_radius' not in featurizer_config:
            errors.append("featurizer.cluster_radius is required for CR method")
        elif method == FeaturizerMethod.NTH_NN.value and 'neighbor_depth' not in featurizer_config:
            errors.append("featurizer.neighbor_depth is required for nth-NN method")
        elif method and method not in [FeaturizerMethod.CR.value, FeaturizerMethod.NTH_NN.value]:
            errors.append(f"featurizer.method must be '{FeaturizerMethod.CR.value}' or '{FeaturizerMethod.NTH_NN.value}'")
        
        return errors
    
    @staticmethod
    def _validate_logging_section(config: Dict[str, Any], is_prediction_config: bool) -> List[str]:
        """Validate logging section."""
        errors = []
        
        if 'logging' not in config:
            return errors
        
        logging_config = config['logging']
        
        # For prediction configs, only log_dir is required
        if is_prediction_config:
            required_keys = ['log_dir']
        else:
            required_keys = ['log_dir', 'monitor', 'mode']
        
        for key in required_keys:
            if key not in logging_config:
                errors.append(f"logging.{key} is required")
        
        # Validate mode
        mode = logging_config.get('mode')
        if mode and mode not in ['min', 'max']:
            errors.append("logging.mode must be 'min' or 'max'")
        
        return errors
    
    @staticmethod
    def _validate_task_settings(config: Dict[str, Any]) -> List[str]:
        """Validate task-specific configuration settings."""
        warnings_list = []
        
        model_config = config.get('model', {})
        logging_config = config.get('logging', {})
        task = model_config.get('task', TaskType.REGRESSION.value)
        monitor = logging_config.get('monitor', '')
        
        if task == TaskType.CLASSIFICATION.value:
            if 'mae' in monitor.lower():
                warnings_list.append(
                    "For classification tasks, consider using 'val_acc' or 'val_f1' "
                    "instead of MAE-based metrics for monitoring"
                )
            if model_config.get('n_classes', 0) <= 1:
                warnings_list.append(
                    "Classification task should have n_classes > 1. "
                    "Consider setting model.n_classes appropriately"
                )
        elif task == TaskType.REGRESSION.value:
            if 'acc' in monitor.lower():
                warnings_list.append(
                    "For regression tasks, consider using 'val_mae', 'val_mse', or 'val_loss' "
                    "instead of accuracy-based metrics for monitoring"
                )
        
        return warnings_list
    
    @staticmethod
    def _validate_training_params(config: Dict[str, Any]) -> List[str]:
        """Validate training parameters for common issues."""
        warnings_list = []
        
        # Skip training validation for prediction-only configs
        if ConfigValidator._is_prediction_only_config(config):
            return warnings_list
        
        training_config = config.get('training', {})
        
        # Validate epochs
        warnings_list.extend(ConfigValidator._validate_epochs(training_config))
        
        # Validate learning rate
        warnings_list.extend(ConfigValidator._validate_learning_rate(training_config))
        
        # Validate batch size
        warnings_list.extend(ConfigValidator._validate_batch_size(training_config))
        
        # Validate early stopping
        warnings_list.extend(ConfigValidator._validate_early_stopping(training_config))
        
        return warnings_list
    
    @staticmethod
    def _validate_epochs(training_config: Dict[str, Any]) -> List[str]:
        """Validate epoch configuration."""
        warnings_list = []
        epochs = training_config.get('epochs', 0)
        
        if epochs > ConfigConstants.HIGH_EPOCHS_THRESHOLD:
            warnings_list.append(
                f"High epoch count ({epochs}). Consider enabling early stopping "
                "to prevent overfitting and reduce training time"
            )
        elif epochs < ConfigConstants.LOW_EPOCHS_THRESHOLD:
            warnings_list.append(
                f"Very low epoch count ({epochs}). Training might be insufficient"
            )
        
        return warnings_list
    
    @staticmethod
    def _validate_learning_rate(training_config: Dict[str, Any]) -> List[str]:
        """Validate learning rate configuration."""
        warnings_list = []
        lr = training_config.get('lr', 0)
        
        if lr > ConfigConstants.HIGH_LR_THRESHOLD:
            warnings_list.append(
                f"High learning rate ({lr}). Consider reducing for training stability"
            )
        elif lr < ConfigConstants.LOW_LR_THRESHOLD:
            warnings_list.append(
                f"Very low learning rate ({lr}). Training might be extremely slow"
            )
        
        return warnings_list
    
    @staticmethod
    def _validate_batch_size(training_config: Dict[str, Any]) -> List[str]:
        """Validate batch size configuration."""
        warnings_list = []
        batch_size = training_config.get('batch_size', 0)
        
        if batch_size > ConfigConstants.LARGE_BATCH_THRESHOLD:
            warnings_list.append(
                f"Large batch size ({batch_size}). Ensure sufficient memory and "
                "consider adjusting learning rate accordingly"
            )
        elif batch_size < ConfigConstants.SMALL_BATCH_THRESHOLD:
            warnings_list.append(
                f"Small batch size ({batch_size}). Training might be unstable"
            )
        
        return warnings_list
    
    @staticmethod
    def _validate_early_stopping(training_config: Dict[str, Any]) -> List[str]:
        """Validate early stopping configuration."""
        warnings_list = []
        early_stopping = training_config.get('early_stopping', {})
        
        if isinstance(early_stopping, dict) and early_stopping.get('enabled', False):
            epochs = training_config.get('epochs', 0)
            patience = early_stopping.get('patience', 0)
            
            if patience > epochs // 4:
                warnings_list.append(
                    f"Early stopping patience ({patience}) is high relative to total epochs ({epochs}). "
                    "Consider reducing patience for better efficiency"
                )
        
        return warnings_list
    
    @staticmethod
    def _validate_data_params(config: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Validate data configuration parameters."""
        warnings_list = []
        errors = []
        
        data_config = config.get('data', {})
        
        # Get data ratios with defaults
        train_ratio = data_config.get('train_ratio', ConfigConstants.DEFAULT_TRAIN_RATIO)
        val_ratio = data_config.get('val_ratio', ConfigConstants.DEFAULT_VAL_RATIO)
        test_ratio = data_config.get('test_ratio', ConfigConstants.DEFAULT_TEST_RATIO)
        
        # Validate ratio sums based on cross-validation settings
        cv_config = config.get('cross_validation', {})
        if cv_config.get('enabled', False):
            # For k-fold CV, only train_ratio + test_ratio should sum to 1.0
            total_ratio = train_ratio + test_ratio
            if abs(total_ratio - 1.0) > ConfigConstants.MIN_RATIO_SUM_TOLERANCE:
                errors.append(
                    f"For k-fold CV, train_ratio + test_ratio must sum to 1.0, got {total_ratio} "
                    f"(train: {train_ratio}, test: {test_ratio})"
                )
        else:
            # For normal training, all ratios should sum to 1.0
            total_ratio = train_ratio + val_ratio + test_ratio
            if abs(total_ratio - 1.0) > ConfigConstants.MIN_RATIO_SUM_TOLERANCE:
                errors.append(
                    f"Data ratios must sum to 1.0, got {total_ratio} "
                    f"(train: {train_ratio}, val: {val_ratio}, test: {test_ratio})"
                )
        
        # Generate warnings for unusual ratio configurations
        if train_ratio == 0.0:
            warnings_list.append(
                "Train ratio is 0.0. This configuration is suitable only for prediction mode"
            )
        elif train_ratio < 0.5:
            warnings_list.append(
                f"Low train ratio ({train_ratio}). Consider increasing for better model training"
            )
        
        if val_ratio == 0.0 and not cv_config.get('enabled', False):
            warnings_list.append(
                "Validation ratio is 0.0. Model validation and early stopping will be disabled"
            )
        
        return warnings_list, errors
    
    @staticmethod
    def _validate_cv_settings(config: Dict[str, Any]) -> List[str]:
        """Validate cross-validation settings."""
        warnings_list = []
        
        cv_config = config.get('cross_validation', {})
        
        if cv_config.get('enabled', False):
            n_folds = cv_config.get('n_folds', ConfigConstants.DEFAULT_FOLDS)
            
            if n_folds < ConfigConstants.MIN_FOLDS:
                warnings_list.append(
                    f"Very few CV folds ({n_folds}). Consider using at least {ConfigConstants.MIN_FOLDS}-5 folds"
                )
            elif n_folds > ConfigConstants.MAX_FOLDS:
                warnings_list.append(
                    f"Many CV folds ({n_folds}). This will significantly increase training time"
                )
            
            # Check data ratios compatibility with CV
            data_config = config.get('data', {})
            val_ratio = data_config.get('val_ratio', ConfigConstants.DEFAULT_VAL_RATIO)
            
            if val_ratio > 0.0:
                warnings_list.append(
                    "Cross-validation is enabled but validation ratio > 0. "
                    "CV will handle validation internally"
                )
        
        return warnings_list
    
    @staticmethod
    def _validate_slurm_settings(config: Dict[str, Any]) -> List[str]:
        """Validate SLURM configuration settings."""
        warnings_list = []
        
        slurm_config = config.get('slurm', {})
        
        if slurm_config.get('use_slurm', False):
            # Validate memory specification
            mem = slurm_config.get('mem', '')
            if mem and not any(unit in mem.upper() for unit in ConfigConstants.MEMORY_UNITS):
                warnings_list.append(
                    f"SLURM memory specification '{mem}' should include units ({', '.join(ConfigConstants.MEMORY_UNITS)})"
                )
            
            # Validate GPU resources alignment
            gres = slurm_config.get('gres', '')
            training_devices = config.get('training', {}).get('devices', 1)
            
            if gres and 'gpu:' in gres:
                try:
                    gpu_count = int(gres.split('gpu:')[1].split(',')[0])
                    if gpu_count != training_devices:
                        warnings_list.append(
                            f"SLURM GPU count ({gpu_count}) doesn't match training devices ({training_devices})"
                        )
                except (ValueError, IndexError):
                    warnings_list.append(f"Could not parse SLURM gres specification: '{gres}'")
        
        return warnings_list


class ConfigTemplateGenerator:
    """Handles configuration template generation."""
    
    @staticmethod
    def create_template(template_type: str, output_path: Optional[str] = None) -> str:
        """
        Create configuration templates for different use cases.
        
        Args:
            template_type: Type of template ('regression', 'classification', 'prediction')
            output_path: Path to save template. If None, returns as string
            
        Returns:
            Path to saved template or YAML content
        """
        try:
            template_enum = TemplateType(template_type)
        except ValueError:
            valid_types = [t.value for t in TemplateType]
            raise ValueError(f"Unknown template type: {template_type}. Valid types: {valid_types}")
        
        # Generate template based on type
        if template_enum == TemplateType.REGRESSION:
            config = ConfigTemplateGenerator._get_regression_template()
        elif template_enum == TemplateType.CLASSIFICATION:
            config = ConfigTemplateGenerator._get_classification_template()
        elif template_enum == TemplateType.PREDICTION:
            config = ConfigTemplateGenerator._get_prediction_template()
        
        # Add template metadata
        config['_template_info'] = {
            'type': template_type,
            'created_by': 'cgnet-cli',
            'description': ConfigTemplateGenerator._get_template_description(template_type)
        }
        
        if output_path:
            ConfigManager.save_config(config, output_path)
            logger.info(f"Template saved to: {output_path}")
            return output_path
        else:
            return yaml.dump(config, default_flow_style=False, indent=2)
    
    @staticmethod
    def _get_regression_template() -> Dict[str, Any]:
        """Get optimized template for regression tasks."""
        template = ConfigManager.get_default_config()
        
        # Optimize for regression
        template['model']['task'] = TaskType.REGRESSION.value
        template['training']['epochs'] = ConfigConstants.DEFAULT_EPOCHS
        template['training']['lr'] = ConfigConstants.DEFAULT_LEARNING_RATE
        template['training']['early_stopping']['enabled'] = False
        template['training']['early_stopping']['monitor'] = 'val_mae'
        template['training']['early_stopping']['mode'] = 'min'
        template['logging']['monitor'] = 'val_mae'
        template['logging']['mode'] = 'min'
        
        return template
    
    @staticmethod
    def _get_classification_template() -> Dict[str, Any]:
        """Get optimized template for classification tasks."""
        template = ConfigManager.get_default_config()
        
        # Optimize for classification
        template['model']['task'] = TaskType.CLASSIFICATION.value
        template['model']['n_classes'] = 2
        template['training']['epochs'] = 200
        template['training']['lr'] = ConfigConstants.DEFAULT_LEARNING_RATE
        template['training']['early_stopping']['enabled'] = False
        template['training']['early_stopping']['monitor'] = 'val_acc'
        template['training']['early_stopping']['mode'] = 'max'
        template['cross_validation']['stratified'] = True
        template['logging']['monitor'] = 'val_acc'
        template['logging']['mode'] = 'max'
        
        return template
    
    @staticmethod
    def _get_prediction_template() -> Dict[str, Any]:
        """Get simplified template for prediction tasks."""
        return {
            'experiment': {
                'name': 'cgnet_prediction',
                'seed': ConfigConstants.DEFAULT_SEED
            },
            'data': {
                'path': 'prediction_dataset',
                'train_ratio': ConfigConstants.DEFAULT_TRAIN_RATIO,
                'val_ratio': ConfigConstants.DEFAULT_VAL_RATIO,
                'test_ratio': ConfigConstants.DEFAULT_TEST_RATIO,
                'save_dir': 'prediction_graph_dataset',
                'force_reload': False,
                'filter_isolated_nodes': True
            },
            'model': {
                'in_node_dim': 92,
                'hidden_node_dim': 64,
                'in_edge_dim': 41,
                'predictor_hidden_dim': 128,
                'num_conv': 3,
                'n_h': 2,
                'n_tasks': 1,
                'task': TaskType.REGRESSION.value,
                'n_classes': 2
            },
            'training': {
                'epochs': 1,
                'batch_size': 128,
                'lr': ConfigConstants.DEFAULT_LEARNING_RATE
            },
            'prediction': {
                'model_path': None,
                'batch_size': 128,
                'num_workers': 4,
                'output_file': 'predictions.csv',
                'save_probabilities': True,
                'save_raw_outputs': True
            },
            'featurizer': {
                'method': FeaturizerMethod.CR.value,
                'neighbor_radius': 8.0,
                'max_neighbors': 12,
                # CR method parameters
                'cluster_radius': 7.0,
                # nth-NN method parameters
                'neighbor_depth': 2,
                'neighbor_mult': 1.15,
                'max_distance_factor': 2.0,
                'small_lattice_threshold': 5.0,
                'enable_smart_images': True,
                'conservative_small_lattice': False,
                # Common parameters
                'max_cluster_nodes': None,
                'tag': 2,
                'step': 0.2,
                'with_pseudonodes': True
            },
            'device': {
                'accelerator': 'auto',
                'devices': 1,
                'precision': 32
            },
            'logging': {
                'log_dir': 'logs/prediction',
                'monitor': 'val_loss',
                'mode': 'min'
            },
            'slurm': {
                'use_slurm': False,
                'partition': 'Normal',
                'nodes': 1,
                'cpus_per_task': 4,
                'mem': '16GB',
                'gres': 'gpu:1',
                'time': '04:00:00',
                'job_name': 'cgnet_prediction',
                'output': 'slurm_logs/cgnet_prediction_%j.out',
                'error': 'slurm_logs/cgnet_prediction_%j.err'
            },
            'advanced': {
                'uncertainty_estimation': False,
                'ensemble_prediction': False,
                'ensemble_models': [],
                'max_cluster_nodes': None
            }
        }
    
    @staticmethod
    def _get_template_description(template_type: str) -> str:
        """Get description for template type."""
        descriptions = {
            TemplateType.REGRESSION.value: 'Optimized configuration for regression tasks with MAE monitoring',
            TemplateType.CLASSIFICATION.value: 'Optimized configuration for classification tasks with accuracy monitoring',
            TemplateType.PREDICTION.value: 'Optimized configuration for inference on new data'
        }
        return descriptions.get(template_type, 'Custom configuration template')
    
    @staticmethod
    def list_available_templates() -> Dict[str, str]:
        """List all available templates and their descriptions."""
        return {
            template.value: ConfigTemplateGenerator._get_template_description(template.value)
            for template in TemplateType
        }


class ConfigOptimizer:
    """Provides configuration optimization suggestions."""
    
    @staticmethod
    def suggest_optimizations(config: Dict[str, Any]) -> List[str]:
        """
        Suggest configuration optimizations based on current settings.
        
        Args:
            config: Configuration dictionary to analyze
            
        Returns:
            List of optimization suggestions
        """
        suggestions = []
        
        model_config = config.get('model', {})
        training_config = config.get('training', {})
        
        # Task-specific optimizations
        suggestions.extend(ConfigOptimizer._suggest_task_optimizations(config))
        
        # Training optimizations
        suggestions.extend(ConfigOptimizer._suggest_training_optimizations(training_config))
        
        # Performance optimizations
        suggestions.extend(ConfigOptimizer._suggest_performance_optimizations(training_config))
        
        return suggestions
    
    @staticmethod
    def _suggest_task_optimizations(config: Dict[str, Any]) -> List[str]:
        """Suggest task-specific optimizations."""
        suggestions = []
        
        model_config = config.get('model', {})
        task = model_config.get('task', TaskType.REGRESSION.value)
        
        if task == TaskType.CLASSIFICATION.value:
            cv_config = config.get('cross_validation', {})
            if not cv_config.get('stratified', False):
                suggestions.append(
                    "For classification tasks, consider enabling stratified cross-validation "
                    "to ensure balanced class distribution across folds"
                )
        
        return suggestions
    
    @staticmethod
    def _suggest_training_optimizations(training_config: Dict[str, Any]) -> List[str]:
        """Suggest training-related optimizations."""
        suggestions = []
        
        if not training_config.get('early_stopping', {}).get('enabled', False):
            suggestions.append(
                "Consider enabling early stopping to prevent overfitting and reduce training time"
            )
        
        return suggestions
    
    @staticmethod
    def _suggest_performance_optimizations(training_config: Dict[str, Any]) -> List[str]:
        """Suggest performance-related optimizations."""
        suggestions = []
        
        num_workers = training_config.get('num_workers', 0)
        if num_workers <= 1:
            suggestions.append(
                "Consider increasing num_workers (e.g., 4-8) for faster data loading"
            )
        
        return suggestions


class ConfigManager:
    """Unified configuration manager for CG-NET training pipeline.
    
    This class centralizes all configuration-related operations including:
    - Loading and saving configurations
    - Validation and error checking
    - Template generation
    - CLI parameter override handling
    - Default configuration management
    """
    
    # =====================================================
    # Core Configuration Operations
    # =====================================================
    
    @staticmethod
    def load_config(config_path: str, silent_validation: bool = False) -> Dict[str, Any]:
        """
        Load and validate configuration from YAML file.
        
        Args:
            config_path: Path to the YAML configuration file
            silent_validation: If True, suppress validation output
            
        Returns:
            Validated configuration dictionary
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist
            ValueError: If configuration validation fails
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format in {config_path}: {e}")
        
        # Validate configuration
        warnings_list, errors = ConfigValidator.validate_config(config)
        
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
            raise ValueError(error_msg)
        
        if not silent_validation:
            if warnings_list:
                logger.warning("Configuration warnings:")
                for warning in warnings_list:
                    logger.warning(f"  {warning}")
            else:
                logger.info("Configuration validation passed")
        
        return config
    
    @staticmethod
    def save_config(config: Dict[str, Any], save_path: str) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            config: Configuration dictionary to save
            save_path: Path where to save the configuration
            
        Raises:
            OSError: If file cannot be created or written
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(save_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to: {save_path}")
        except Exception as e:
            raise OSError(f"Failed to save configuration to {save_path}: {e}")
    
    @staticmethod
    def update_config(config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update configuration with new values using deep merge.
        
        Args:
            config: Original configuration
            updates: Dictionary containing configuration updates
            
        Returns:
            Updated configuration dictionary
        """
        def deep_merge(target: dict, source: dict) -> dict:
            """Recursively merge two dictionaries."""
            for key, value in source.items():
                if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                    target[key] = deep_merge(target[key], value)
                else:
                    target[key] = value
            return target
        
        updated_config = deep_merge(copy.deepcopy(config), updates)
        logger.debug(f"Configuration updated with {len(updates)} changes")
        
        return updated_config
    
    # =====================================================
    # Validation
    # =====================================================
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """
        Comprehensive configuration validation.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            Tuple of (warnings, errors) - Lists of warning and error messages
        """
        return ConfigValidator.validate_config(config)
    
    # =====================================================
    # Template Management
    # =====================================================
    
    @staticmethod
    def create_template(template_type: str, output_path: Optional[str] = None) -> str:
        """
        Create configuration templates for different use cases.
        
        Args:
            template_type: Type of template ('regression', 'classification', 'prediction')
            output_path: Path to save template. If None, returns as string
            
        Returns:
            Path to saved template or YAML content
        """
        return ConfigTemplateGenerator.create_template(template_type, output_path)
    
    @staticmethod
    def list_available_templates() -> Dict[str, str]:
        """List all available templates and their descriptions."""
        return ConfigTemplateGenerator.list_available_templates()
    
    # =====================================================
    # Optimization Suggestions
    # =====================================================
    
    @staticmethod
    def suggest_optimizations(config: Dict[str, Any]) -> List[str]:
        """Suggest configuration optimizations based on current settings."""
        return ConfigOptimizer.suggest_optimizations(config)
    
    # =====================================================
    # Default Configuration
    # =====================================================
    
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """
        Get default configuration template.
        
        Returns:
            Default configuration dictionary with sensible defaults
        """
        return {
            'experiment': {
                'name': 'cgnet_experiment',
                'seed': ConfigConstants.DEFAULT_SEED
            },
            'data': {
                'path': 'raw_dataset',
                'train_ratio': ConfigConstants.DEFAULT_TRAIN_RATIO,
                'val_ratio': ConfigConstants.DEFAULT_VAL_RATIO,
                'test_ratio': ConfigConstants.DEFAULT_TEST_RATIO,
                'save_dir': 'graph_dataset',
                'force_reload': False,
                'filter_isolated_nodes': True
            },
            'model': {
                'in_node_dim': 92,
                'hidden_node_dim': 64,
                'in_edge_dim': 41,
                'predictor_hidden_dim': 128,
                'num_conv': 3,
                'n_h': 2,
                'n_tasks': 1,
                'task': TaskType.REGRESSION.value,
                'n_classes': 2
            },
            'training': {
                'epochs': ConfigConstants.DEFAULT_EPOCHS,
                'batch_size': ConfigConstants.DEFAULT_BATCH_SIZE,
                'lr': ConfigConstants.DEFAULT_LEARNING_RATE,
                'num_workers': 0,
                'tmax': ConfigConstants.DEFAULT_EPOCHS,
                'devices': 1,
                'strategy': None,
                'early_stopping': {
                    'enabled': False,
                    'monitor': 'val_loss',
                    'patience': 10,
                    'mode': 'min',
                    'min_delta': 0.0,
                    'verbose': True
                }
            },
            'cross_validation': {
                'enabled': False,
                'n_folds': ConfigConstants.DEFAULT_FOLDS,
                'stratified': False,
                'shuffle': True,
                'save_fold_results': True,
                'aggregate_results': True
            },
            'featurizer': {
                'method': FeaturizerMethod.CR.value,
                'neighbor_radius': 8.0,
                'max_neighbors': 12,
                # CR method parameters
                'cluster_radius': 7.0,
                # nth-NN method parameters
                'neighbor_depth': 2,
                'neighbor_mult': 1.15,
                'max_distance_factor': 2.0,
                'small_lattice_threshold': 5.0,
                'enable_smart_images': True,
                'conservative_small_lattice': False,
                # Common parameters
                'max_cluster_nodes': None,
                'tag': 2,
                'step': 0.2,
                'with_pseudonodes': True
            },
            'logging': {
                'log_dir': 'logs',
                'monitor': 'val_mae',
                'mode': 'min',
                'save_top_k': 1
            },
            'slurm': {
                'use_slurm': False,
                'partition': 'Normal',
                'nodes': 1,
                'ntasks_per_node': 1,
                'cpus_per_task': 4,
                'mem': '32GB',
                'gres': 'gpu:1',
                'time': '24:00:00',
                'job_name': 'cgnet_experiment',
                'account': None,
                'output': 'slurm_logs/cgnet_%j.out',
                'error': 'slurm_logs/cgnet_%j.err',
                'qos': None,
                'constraint': None,
                'exclude': None,
                'mail_type': None,
                'mail_user': None,
                'nice': None,
                'gpus_per_node': 1
            },
            'device': {
                'accelerator': 'auto',
                'precision': 32,
                'deterministic': False,
                'benchmark': True
            }
        }
    
    # =====================================================
    # CLI Support Methods
    # =====================================================
    
    @staticmethod
    def load_or_create_config(config_path: Optional[str], use_defaults: bool) -> Dict[str, Any]:
        """
        Load configuration from file or create default configuration.
        
        Args:
            config_path: Path to configuration file
            use_defaults: Whether to create default config if not found
            
        Returns:
            Configuration dictionary
            
        Raises:
            ValueError: If neither config_path nor use_defaults is specified
        """
        if use_defaults:
            if config_path is None:
                config_path = ConfigConstants.DEFAULT_CONFIG_NAME
            
            # Create default config if it doesn't exist
            if not os.path.exists(config_path):
                logger.info(f"Creating default configuration file: {config_path}")
                default_config = ConfigManager.get_default_config()
                ConfigManager.save_config(default_config, config_path)
                logger.info(f"Default configuration saved to {config_path}")
                logger.info("Please review and modify the configuration as needed.")
                return default_config
            else:
                logger.info(f"Loading existing configuration: {config_path}")
                return ConfigManager.load_config(config_path)
        
        elif config_path:
            return ConfigManager.load_config(config_path)
        
        else:
            raise ValueError(
                "Either --config <path> or --use-defaults must be specified. "
                "Use --help for more information."
            )
    
    @staticmethod
    def print_config_summary(config: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None) -> None:
        """
        Print a summary of the configuration and any CLI overrides.
        
        Args:
            config: Final configuration dictionary
            overrides: CLI overrides applied to configuration
        """
        print("=" * 60)
        print("CONFIGURATION SUMMARY")
        print("=" * 60)
        
        # Print key configuration details
        print(f"Experiment: {config.get('experiment', {}).get('name', 'Unknown')}")
        print(f"Task: {config.get('model', {}).get('task', 'Unknown')}")
        print(f"Data path: {config.get('data', {}).get('path', 'Unknown')}")
        print(f"Epochs: {config.get('training', {}).get('epochs', 'N/A')}")
        print(f"Batch size: {config.get('training', {}).get('batch_size', 'N/A')}")
        print(f"Learning rate: {config.get('training', {}).get('lr', 'N/A')}")
        
        cv_config = config.get('cross_validation', {})
        if cv_config.get('enabled', False):
            print(f"Cross-validation: {cv_config.get('n_folds', 'Unknown')} folds")
        
        slurm_config = config.get('slurm', {})
        if slurm_config.get('use_slurm', False):
            print(f"SLURM: Enabled (partition: {slurm_config.get('partition', 'Unknown')})")
        
        # Print CLI overrides if any
        if overrides:
            print("\nCLI Overrides Applied:")
            print("-" * 30)
            ConfigManager._print_nested_dict(overrides, indent=2)
        
        print("=" * 60)
    
    @staticmethod
    def _print_nested_dict(d: Dict[str, Any], indent: int = 0) -> None:
        """
        Helper function to print nested dictionary.
        
        Args:
            d: Dictionary to print
            indent: Indentation level
        """
        for key, value in d.items():
            if isinstance(value, dict):
                print(" " * indent + f"{key}:")
                ConfigManager._print_nested_dict(value, indent + 2)
            else:
                print(" " * indent + f"{key}: {value}") 