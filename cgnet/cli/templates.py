#!/usr/bin/env python
"""Configuration template management for CG-NET CLI."""

import yaml
from typing import Dict, Any

from ..trainers import ConfigManager


class ConfigTemplateManager:
    """Manages configuration templates for different use cases."""
    
    @staticmethod
    def create_template(template_type: str, output_path: str = None) -> str:
        """
        Create configuration templates for different use cases.
        
        Parameters
        ----------
        template_type : str
            Type of template: 'regression', 'classification', 'prediction'
        output_path : str, optional
            Path to save template. If None, returns as string
            
        Returns
        -------
        str
            Path to saved template or YAML content
        """
        if template_type == 'regression':
            config = ConfigTemplateManager._get_regression_template()
        elif template_type == 'classification':
            config = ConfigTemplateManager._get_classification_template()
        elif template_type == 'prediction':
            config = ConfigTemplateManager._get_prediction_template()
        else:
            raise ValueError(f"Unknown template type: {template_type}")
        
        # Add template metadata
        config['_template_info'] = {
            'type': template_type,
            'created_by': 'cgnet-cli',
            'description': ConfigTemplateManager._get_template_description(template_type)
        }
        
        if output_path:
            ConfigManager.save_config(config, output_path)
            return output_path
        else:
            return yaml.dump(config, default_flow_style=False, indent=2)
    
    @staticmethod
    def _get_regression_template() -> Dict[str, Any]:
        """Get optimized template for regression tasks."""
        template = ConfigManager.get_default_config()
        
        # Optimize for regression
        template['model']['task'] = 'regression'
        template['training']['epochs'] = 300
        template['training']['lr'] = 0.005
        template['training']['early_stopping']['enabled'] = True
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
        template['model']['task'] = 'classification'
        template['model']['n_classes'] = 2
        template['training']['epochs'] = 200
        template['training']['lr'] = 0.001
        template['training']['early_stopping']['enabled'] = True
        template['training']['early_stopping']['monitor'] = 'val_acc'
        template['training']['early_stopping']['mode'] = 'max'
        template['cross_validation']['stratified'] = True
        template['logging']['monitor'] = 'val_acc'
        template['logging']['mode'] = 'max'
        
        return template
    
    @staticmethod
    def _get_prediction_template() -> Dict[str, Any]:
        """Get optimized template for prediction tasks."""
        template = ConfigManager.get_default_config()
        
        # Optimize for prediction
        template['data']['train_ratio'] = 0.0
        template['data']['val_ratio'] = 0.0
        template['data']['test_ratio'] = 1.0
        template['training']['batch_size'] = 128
        template['training']['num_workers'] = 4
        template['logging']['save_top_k'] = 0
        
        # Add prediction-specific settings
        template['prediction'] = {
            'model_path': None,
            'output_file': 'predictions.csv',
            'save_probabilities': True,
            'save_raw_outputs': True,
            'uncertainty_estimation': False
        }
        
        return template
    
    @staticmethod
    def _get_template_description(template_type: str) -> str:
        """Get description for template type."""
        descriptions = {
            'regression': 'Optimized configuration for regression tasks with MAE monitoring',
            'classification': 'Optimized configuration for classification tasks with accuracy monitoring',
            'prediction': 'Optimized configuration for inference on new data'
        }
        return descriptions.get(template_type, 'Custom configuration template')
    
    @staticmethod
    def list_available_templates() -> Dict[str, str]:
        """List all available templates and their descriptions."""
        return {
            'regression': ConfigTemplateManager._get_template_description('regression'),
            'classification': ConfigTemplateManager._get_template_description('classification'),
            'prediction': ConfigTemplateManager._get_template_description('prediction')
        } 