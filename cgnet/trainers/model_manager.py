#!/usr/bin/env python
"""Model management for CG-NET training pipeline."""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import pytorch_lightning as pl

from ..models import CGNET


class ModelManager:
    """Manages model creation, loading, and checkpointing for CG-NET training."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ModelManager with configuration.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary
        """
        self.config = config
        self.model = None
    
    def create_model(self, in_edge_dim: int) -> CGNET:
        """Create CG-NET model based on configuration."""
        # Update edge dimension in config
        self.config['model']['in_edge_dim'] = in_edge_dim
        
        model_config = self.config['model']
        training_config = self.config['training']
        
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
            lr=training_config['lr'],
            tmax=training_config.get('tmax', training_config['epochs'])
        )
        
        self.model = model
        return model
    
    def load_model_from_checkpoint(self, checkpoint_path: str) -> CGNET:
        """
        Load model from checkpoint.
        
        Parameters
        ----------
        checkpoint_path : str
            Path to model checkpoint
            
        Returns
        -------
        CGNET
            Loaded model
        """
        model_config = self.config['model']
        training_config = self.config['training']
        
        model = CGNET.load_from_checkpoint(
            checkpoint_path,
            in_node_dim=model_config['in_node_dim'],
            hidden_node_dim=model_config['hidden_node_dim'],
            in_edge_dim=model_config['in_edge_dim'],
            predictor_hidden_dim=model_config['predictor_hidden_dim'],
            n_conv=model_config['num_conv'],
            n_h=model_config['n_h'],
            n_tasks=model_config['n_tasks'],
            task=model_config['task'],
            n_classes=model_config['n_classes'],
            lr=training_config['lr'],
            tmax=training_config.get('tmax', training_config['epochs'])
        )
        
        self.model = model
        return model
    
    def get_model_summary(self) -> str:
        """
        Get a summary of the model architecture.
        
        Returns
        -------
        str
            Model summary string
        """
        if self.model is None:
            return "No model loaded"
        
        model_config = self.config['model']
        summary = f"""
CG-NET Model Configuration:
  - Task: {model_config['task']}
  - Input node features: {model_config['in_node_dim']}
  - Hidden node features: {model_config['hidden_node_dim']}
  - Input edge features: {model_config.get('in_edge_dim', 'Not set')}
  - Predictor hidden features: {model_config['predictor_hidden_dim']}
  - Number of conv layers: {model_config['num_conv']}
  - Number of heads: {model_config['n_h']}
  - Number of tasks: {model_config['n_tasks']}
"""
        if model_config['task'] == 'classification':
            summary += f"  - Number of classes: {model_config['n_classes']}\n"
        
        # Count parameters
        if self.model is not None:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            summary += f"  - Total parameters: {total_params:,}\n"
            summary += f"  - Trainable parameters: {trainable_params:,}\n"
        
        return summary
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """
        Find the latest model checkpoint.
        
        Returns
        -------
        str or None
            Path to the latest checkpoint, or None if no checkpoints found
        """
        logs_dir = Path(self.config['logging']['log_dir'])
        checkpoint_pattern = "**/*.ckpt"
        
        checkpoints = list(logs_dir.glob(checkpoint_pattern))
        if checkpoints:
            # Sort by modification time and get the latest
            latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
            return str(latest_checkpoint)
        return None
    
    def get_best_checkpoint(self, experiment_name: str = None) -> Optional[str]:
        """
        Find the best model checkpoint based on the monitoring metric.
        
        Parameters
        ----------
        experiment_name : str, optional
            Specific experiment name to search in
            
        Returns
        -------
        str or None
            Path to the best checkpoint, or None if not found
        """
        logs_dir = Path(self.config['logging']['log_dir'])
        
        if experiment_name:
            search_dir = logs_dir / experiment_name
        else:
            search_dir = logs_dir
        
        # Look for checkpoints
        checkpoint_pattern = "**/checkpoints/*.ckpt"
        checkpoints = list(search_dir.glob(checkpoint_pattern))
        
        if not checkpoints:
            return None
        
        # Try to find the best checkpoint based on filename patterns
        monitor_metric = self.config['logging']['monitor']
        mode = self.config['logging']['mode']
        
        best_checkpoint = None
        best_value = None
        
        for checkpoint in checkpoints:
            # Try to extract metric value from filename
            filename = checkpoint.stem
            if monitor_metric.replace('_', '-') in filename or monitor_metric in filename:
                try:
                    # Extract the metric value from filename
                    # Typical format: epoch=02-step=03-val_mae=0.15.ckpt
                    parts = filename.split('-')
                    for part in parts:
                        if monitor_metric.replace('_', '-') in part or monitor_metric in part:
                            value_str = part.split('=')[-1]
                            value = float(value_str)
                            
                            if best_value is None:
                                best_value = value
                                best_checkpoint = checkpoint
                            elif (mode == 'min' and value < best_value) or (mode == 'max' and value > best_value):
                                best_value = value
                                best_checkpoint = checkpoint
                            break
                except (ValueError, IndexError):
                    continue
        
        return str(best_checkpoint) if best_checkpoint else str(checkpoints[0])
    
    def save_model_state(self, save_path: str) -> None:
        """
        Save current model state.
        
        Parameters
        ----------
        save_path : str
            Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save model state dict
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'model_config': self.config['model']
        }, save_path)
        
        print(f"Model saved to: {save_path}")
    
    def load_model_state(self, load_path: str) -> CGNET:
        """
        Load model state from saved file.
        
        Parameters
        ----------
        load_path : str
            Path to load the model from
            
        Returns
        -------
        CGNET
            Loaded model
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        checkpoint = torch.load(load_path, map_location='cpu')
        
        # Update config if saved in checkpoint
        if 'config' in checkpoint:
            self.config.update(checkpoint['config'])
        
        # Create model with loaded config
        model_config = checkpoint.get('model_config', self.config['model'])
        training_config = self.config['training']
        
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
            lr=training_config['lr'],
            tmax=training_config.get('tmax', training_config['epochs'])
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model = model
        print(f"Model loaded from: {load_path}")
        return model
    
    def reset(self) -> None:
        """Reset the model manager state."""
        self.model = None 