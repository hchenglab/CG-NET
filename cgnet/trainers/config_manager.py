#!/usr/bin/env python
"""Configuration management for CG-NET training pipeline."""

import os
import yaml
from typing import Dict, Any


class ConfigManager:
    """Manages configuration loading, validation, and saving for CG-NET training."""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load and validate configuration from YAML file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        ConfigManager.validate_config(config)
        return config
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
        """Validate configuration parameters."""
        required_sections = ['experiment', 'data', 'model', 'training', 'featurizer', 'logging']
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Experiment section
        exp_config = config['experiment']
        if 'name' not in exp_config or not exp_config['name']:
            raise ValueError("experiment.name is required and cannot be empty")
        if 'seed' not in exp_config:
            raise ValueError("experiment.seed is required")
        
        # Data section
        data_config = config['data']
        required_data_keys = ['path', 'train_ratio', 'val_ratio', 'test_ratio']
        for key in required_data_keys:
            if key not in data_config:
                raise ValueError(f"data.{key} is required")
        
        # Validate data ratios
        cv_config = config.get('cross_validation', {})
        if not cv_config.get('enabled', False):
            total_ratio = data_config['train_ratio'] + data_config['val_ratio'] + data_config['test_ratio']
            if abs(total_ratio - 1.0) > 1e-6:
                raise ValueError(f"Data ratios must sum to 1.0, got {total_ratio}")
        else:
            total_ratio = data_config['train_ratio'] + data_config['test_ratio']
            if abs(total_ratio - 1.0) > 1e-6:
                raise ValueError(f"For k-fold CV, train_ratio + test_ratio must sum to 1.0, got {total_ratio}")
        
        # Model section
        model_config = config['model']
        required_model_keys = ['task', 'n_tasks']
        for key in required_model_keys:
            if key not in model_config:
                raise ValueError(f"model.{key} is required")
        
        if model_config['task'] not in ['regression', 'classification']:
            raise ValueError("model.task must be 'regression' or 'classification'")
        
        if model_config['task'] == 'classification' and 'n_classes' not in model_config:
            raise ValueError("model.n_classes is required for classification tasks")
        
        # Training section
        training_config = config['training']
        required_training_keys = ['epochs', 'batch_size', 'lr']
        for key in required_training_keys:
            if key not in training_config:
                raise ValueError(f"training.{key} is required")
        
        # Validate tmax parameter
        if 'tmax' in training_config and training_config['tmax'] is not None:
            if training_config['tmax'] <= 0:
                raise ValueError("training.tmax must be positive")
        
        # Validate device configuration
        if 'devices' in training_config and training_config['devices'] is not None:
            if not isinstance(training_config['devices'], int) or training_config['devices'] < 1:
                raise ValueError("training.devices must be a positive integer")
        
        # Validate early stopping configuration
        if 'early_stopping' in training_config and training_config['early_stopping'].get('enabled', False):
            early_stop_config = training_config['early_stopping']
            if 'patience' in early_stop_config and early_stop_config['patience'] <= 0:
                raise ValueError("training.early_stopping.patience must be positive")
            if 'mode' in early_stop_config and early_stop_config['mode'] not in ['min', 'max']:
                raise ValueError("training.early_stopping.mode must be 'min' or 'max'")
        
        # Featurizer section
        featurizer_config = config['featurizer']
        required_featurizer_keys = ['method', 'neighbor_radius', 'max_neighbors']
        for key in required_featurizer_keys:
            if key not in featurizer_config:
                raise ValueError(f"featurizer.{key} is required")
        
        if featurizer_config['method'] not in ['CR', 'nth-NN']:
            raise ValueError("featurizer.method must be 'CR' or 'nth-NN'")
        
        if featurizer_config['method'] == 'CR' and 'cluster_radius' not in featurizer_config:
            raise ValueError("featurizer.cluster_radius is required for CR method")
        
        if featurizer_config['method'] == 'nth-NN' and 'neighbor_depth' not in featurizer_config:
            raise ValueError("featurizer.neighbor_depth is required for nth-NN method")
        
        # Logging section
        logging_config = config['logging']
        required_logging_keys = ['log_dir', 'monitor', 'mode']
        for key in required_logging_keys:
            if key not in logging_config:
                raise ValueError(f"logging.{key} is required")
        
        if logging_config['mode'] not in ['min', 'max']:
            raise ValueError("logging.mode must be 'min' or 'max'")
        
        print("✓ Configuration validation passed")
    
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """Get default configuration template."""
        return {
            'experiment': {
                'name': 'cgnet_experiment',
                'seed': 42
            },
            'data': {
                'path': 'raw_dataset',
                'train_ratio': 0.8,
                'val_ratio': 0.1,
                'test_ratio': 0.1,
                'save_cache': True,
                'save_dir': 'graph_dataset',
                'clean_cache': False,
                'use_parallel': False,
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
                'task': 'regression',
                'n_classes': 2
            },
            'training': {
                'epochs': 300,
                'batch_size': 64,
                'lr': 0.001,
                'num_workers': 0,
                'tmax': 300,
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
                'n_folds': 5,
                'stratified': False,
                'shuffle': True,
                'save_fold_results': True,
                'aggregate_results': True
            },
            'featurizer': {
                'method': 'CR',
                'neighbor_radius': 8.0,
                'max_neighbors': 12,
                'cluster_radius': 10.0,
                'neighbor_depth': 2,
                'max_cluster_nodes': None,
                'neighbor_mult': 1.15,
                'tag': 2,
                'step': 0.2,
                'with_pseudonodes': True,
                'max_distance_factor': 2.0,
                'small_lattice_threshold': 5.0,
                'enable_smart_images': True,
                'conservative_small_lattice': False
            },
            'logging': {
                'log_dir': 'logs',
                'monitor': 'val_mae',
                'mode': 'min',
                'save_top_k': 3
            },
            'slurm': {
                'use_slurm': False,
                'partition': 'gpu',
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
                'gpus_per_node': None
            },
            'device': {
                'accelerator': 'auto',
                'precision': 32,
                'deterministic': False,
                'benchmark': True
            }
        }
    
    @staticmethod
    def save_config(config: Dict[str, Any], save_path: str) -> None:
        """Save configuration to YAML file."""
        dir_path = os.path.dirname(save_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        with open(save_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
    
    @staticmethod
    def update_config(config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update configuration with new values.
        
        Parameters
        ----------
        config : dict
            Original configuration
        updates : dict
            Dictionary containing configuration updates
            
        Returns
        -------
        dict
            Updated configuration
        """
        def update_nested_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = update_nested_dict(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        updated_config = update_nested_dict(config.copy(), updates)
        return updated_config 