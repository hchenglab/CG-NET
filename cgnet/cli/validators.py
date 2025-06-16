#!/usr/bin/env python
"""Configuration validation utilities for CG-NET CLI."""

from typing import Dict, Any, List, Tuple, Optional
import warnings


class ConfigValidator:
    """Validates CG-NET configurations for potential issues and optimization opportunities."""
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """
        Comprehensive configuration validation.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary to validate
            
        Returns
        -------
        Tuple[List[str], List[str]]
            (warnings, errors) - Lists of warning and error messages
        """
        warnings_list = []
        errors = []
        
        # Validate basic structure
        structure_warnings, structure_errors = ConfigValidator._validate_structure(config)
        warnings_list.extend(structure_warnings)
        errors.extend(structure_errors)
        
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
        
        required_sections = ['experiment', 'data', 'model', 'training', 'logging']
        for section in required_sections:
            if section not in config:
                errors.append(f"Missing required configuration section: '{section}'")
        
        return warnings_list, errors
    
    @staticmethod
    def _validate_task_settings(config: Dict[str, Any]) -> List[str]:
        """Validate task-specific configuration settings."""
        warnings_list = []
        
        model_config = config.get('model', {})
        logging_config = config.get('logging', {})
        task = model_config.get('task', 'regression')
        
        # Task-specific monitoring validation
        monitor = logging_config.get('monitor', '')
        
        if task == 'classification':
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
                
        elif task == 'regression':
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
        
        training_config = config.get('training', {})
        
        # Epochs validation
        epochs = training_config.get('epochs', 0)
        if epochs > 1000:
            warnings_list.append(
                f"High epoch count ({epochs}). Consider enabling early stopping "
                "to prevent overfitting and reduce training time"
            )
        elif epochs < 10:
            warnings_list.append(
                f"Very low epoch count ({epochs}). Training might be insufficient"
            )
        
        # Learning rate validation
        lr = training_config.get('lr', 0)
        if lr > 0.1:
            warnings_list.append(
                f"High learning rate ({lr}). Consider reducing for training stability"
            )
        elif lr < 1e-6:
            warnings_list.append(
                f"Very low learning rate ({lr}). Training might be extremely slow"
            )
        
        # Batch size validation
        batch_size = training_config.get('batch_size', 0)
        if batch_size > 512:
            warnings_list.append(
                f"Large batch size ({batch_size}). Ensure sufficient memory and "
                "consider adjusting learning rate accordingly"
            )
        elif batch_size < 8:
            warnings_list.append(
                f"Small batch size ({batch_size}). Training might be unstable"
            )
        
        # Early stopping validation
        early_stopping = training_config.get('early_stopping', {})
        if isinstance(early_stopping, dict) and early_stopping.get('enabled', False):
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
        
        # Data ratio validation
        train_ratio = data_config.get('train_ratio', 0.8)
        val_ratio = data_config.get('val_ratio', 0.1)
        test_ratio = data_config.get('test_ratio', 0.1)
        
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            errors.append(
                f"Data ratios must sum to 1.0, got {total_ratio} "
                f"(train: {train_ratio}, val: {val_ratio}, test: {test_ratio})"
            )
        
        if train_ratio == 0.0:
            warnings_list.append(
                "Train ratio is 0.0. This configuration is suitable only for prediction mode"
            )
        elif train_ratio < 0.5:
            warnings_list.append(
                f"Low train ratio ({train_ratio}). Consider increasing for better model training"
            )
        
        if val_ratio == 0.0:
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
            n_folds = cv_config.get('n_folds', 5)
            
            if n_folds < 3:
                warnings_list.append(
                    f"Very few CV folds ({n_folds}). Consider using at least 3-5 folds"
                )
            elif n_folds > 10:
                warnings_list.append(
                    f"Many CV folds ({n_folds}). This will significantly increase training time"
                )
            
            # Check data ratios compatibility with CV
            data_config = config.get('data', {})
            val_ratio = data_config.get('val_ratio', 0.1)
            test_ratio = data_config.get('test_ratio', 0.1)
            
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
            # Check memory setting
            mem = slurm_config.get('mem', '')
            if mem and not any(unit in mem.upper() for unit in ['GB', 'MB', 'TB']):
                warnings_list.append(
                    f"SLURM memory specification '{mem}' should include units (GB, MB, TB)"
                )
            
            # Check GPU resources
            gres = slurm_config.get('gres', '')
            training_devices = config.get('training', {}).get('devices', 1)
            
            if gres and 'gpu:' in gres:
                try:
                    gpu_count = int(gres.split('gpu:')[1].split(',')[0])
                    if gpu_count != training_devices:
                        warnings_list.append(
                            f"SLURM GPU count ({gpu_count}) doesn't match training devices ({training_devices})"
                        )
                except:
                    warnings_list.append(f"Could not parse SLURM gres specification: '{gres}'")
            
            # Check time format
            time_limit = slurm_config.get('time', '')
            if time_limit and ':' not in time_limit:
                warnings_list.append(
                    f"SLURM time format '{time_limit}' should be in HH:MM:SS format"
                )
        
        return warnings_list
    
    @staticmethod
    def suggest_optimizations(config: Dict[str, Any]) -> List[str]:
        """Suggest configuration optimizations based on current settings."""
        suggestions = []
        
        model_config = config.get('model', {})
        training_config = config.get('training', {})
        
        # Task-specific optimizations
        task = model_config.get('task', 'regression')
        if task == 'classification':
            cv_config = config.get('cross_validation', {})
            if not cv_config.get('stratified', False):
                suggestions.append(
                    "For classification tasks, consider enabling stratified cross-validation "
                    "to ensure balanced class distribution across folds"
                )
        
        # Training optimizations
        if not training_config.get('early_stopping', {}).get('enabled', False):
            suggestions.append(
                "Consider enabling early stopping to prevent overfitting and reduce training time"
            )
        
        num_workers = training_config.get('num_workers', 0)
        if num_workers <= 1:
            suggestions.append(
                "Consider increasing num_workers (e.g., 4-8) for faster data loading"
            )
        
        return suggestions 