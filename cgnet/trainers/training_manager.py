#!/usr/bin/env python
"""Training management for CG-NET training pipeline."""

import os
import pickle
import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch_geometric.loader import DataLoader

from .config_manager import ConfigManager
from .model_manager import ModelManager


class TrainingManager:
    """Manages training operations for CG-NET including k-fold cross-validation."""
    
    def __init__(self, config: Dict[str, Any], model_manager: ModelManager):
        """
        Initialize TrainingManager.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary
        model_manager : ModelManager
            Model manager instance
        """
        self.config = config
        self.model_manager = model_manager
        self.trainer = None
    
    def setup_trainer(self, experiment_name: str = None, fold_info: str = None) -> pl.Trainer:
        """Setup PyTorch Lightning trainer with callbacks and logger."""
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        logging_config = self.config['logging']
        
        # Setup checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            monitor=logging_config['monitor'],
            filename="{epoch:02d}-{step:02d}-{" + logging_config['monitor'] + ":.2f}",
            save_top_k=logging_config.get('save_top_k', 1),
            mode=logging_config['mode'],
        )
        
        # Setup logger
        exp_name = experiment_name or self.config['experiment']['name']
        if fold_info:
            exp_name = f"{exp_name}_{fold_info}"
        
        logger = TensorBoardLogger(
            logging_config['log_dir'],
            name=exp_name,
            default_hp_metric=False
        )
        
        # Save config to log directory
        if logger.log_dir:
            ConfigManager.save_config(self.config, os.path.join(logger.log_dir, "config.yml"))
        
        # Setup trainer parameters
        training_config = self.config['training']
        device = "cuda" if torch.cuda.is_available() else "cpu"
        devices = training_config.get('devices', 1)
        strategy = training_config.get('strategy', None)
        
        trainer_kwargs = {
            'max_epochs': training_config['epochs'],
            'accelerator': device,
            'devices': devices,
            'logger': logger,
            'callbacks': [lr_monitor, checkpoint_callback],
        }
        
        # Add early stopping if configured
        if training_config.get('early_stopping', {}).get('enabled', False):
            early_stop_config = training_config['early_stopping']
            early_stopping = EarlyStopping(
                monitor=early_stop_config.get('monitor', logging_config['monitor']),
                patience=early_stop_config.get('patience', 10),
                mode=early_stop_config.get('mode', logging_config['mode']),
                min_delta=early_stop_config.get('min_delta', 0.0),
                verbose=early_stop_config.get('verbose', True)
            )
            trainer_kwargs['callbacks'].append(early_stopping)
        
        if strategy:
            trainer_kwargs['strategy'] = strategy
        
        trainer = pl.Trainer(**trainer_kwargs)
        self.trainer = trainer
        return trainer
    
    def train_model(self, train_loader: DataLoader, val_loader: DataLoader, 
                   experiment_name: str = None, fold_info: str = None) -> Tuple[Any, pl.Trainer]:
        """Train the CG-NET model."""
        print("=" * 50)
        print("TRAINING MODEL")
        print("=" * 50)
        
        # Get edge dimension from sample data
        if hasattr(train_loader.dataset, 'dataset'):
            sample_data = train_loader.dataset.dataset[0]
        else:
            sample_data = train_loader.dataset[0]
        
        in_edge_dim = sample_data.edge_attr.shape[-1] if sample_data.edge_attr is not None else 0
        
        # Create model and trainer
        model = self.model_manager.create_model(in_edge_dim)
        trainer = self.setup_trainer(experiment_name, fold_info)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Training on device: {device}")
        
        # Start training
        trainer.fit(model, train_loader, val_loader)
        
        return model, trainer
    
    def test_model(self, model: Any, test_loader: DataLoader, trainer: pl.Trainer = None) -> None:
        """Test the trained model."""
        print("=" * 50)
        print("TESTING MODEL")
        print("=" * 50)
        
        if trainer is None:
            trainer = self.trainer
        
        if trainer is None:
            raise ValueError("No trainer available. Train model first.")
        
        trainer.test(model, dataloaders=test_loader)
    
    def predict_with_model(self, model: Any, data_loader: DataLoader, trainer: pl.Trainer = None) -> Any:
        """Make predictions using a trained model."""
        print("=" * 50)
        print("MAKING PREDICTIONS")
        print("=" * 50)
        
        if trainer is None:
            trainer = self.trainer
        
        if trainer is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # Force single device for prediction to avoid distributed training
            trainer = pl.Trainer(
                accelerator=device,
                devices=1,  # Use only one device
                strategy="auto",  # Use auto strategy (not distributed)
                enable_progress_bar=True,
                enable_model_summary=False,
                logger=False  # Disable logging for prediction
            )
        
        return trainer.predict(model, dataloaders=data_loader)
    
    def run_kfold_training(self, fold_loaders: List[Tuple[DataLoader, DataLoader]], 
                          test_loader: Optional[DataLoader] = None) -> Dict[str, Any]:
        """Run k-fold cross-validation training."""
        print("=" * 60)
        print("RUNNING K-FOLD CROSS-VALIDATION TRAINING")
        print("=" * 60)
        
        cv_config = self.config.get('cross_validation', {})
        n_folds = len(fold_loaders)
        save_fold_results = cv_config.get('save_fold_results', True)
        
        # Store original experiment name and log directory
        original_exp_name = self.config['experiment']['name']
        original_log_dir = self.config['logging']['log_dir']
        
        # Results storage
        fold_results = []
        fold_models = []
        fold_trainers = []
        
        # Run training for each fold
        for fold_idx, (train_loader, val_loader) in enumerate(fold_loaders):
            print(f"\n{'='*50}")
            print(f"TRAINING FOLD {fold_idx + 1}/{n_folds}")
            print(f"{'='*50}")
            
            # Update experiment name and log directory for this fold
            fold_exp_name = f"{original_exp_name}_fold_{fold_idx + 1}"
            fold_log_dir = os.path.join(original_log_dir, f"fold_{fold_idx + 1}")
            
            # Temporarily update config
            self.config['experiment']['name'] = fold_exp_name
            self.config['logging']['log_dir'] = fold_log_dir
            
            # Reset trainer state for new fold
            self.model_manager.reset()
            self.trainer = None
            
            # Train model for this fold
            model, trainer = self.train_model(
                train_loader, val_loader, 
                experiment_name=original_exp_name,
                fold_info=f"fold_{fold_idx + 1}"
            )
            
            # Store fold results
            fold_models.append(model)
            fold_trainers.append(trainer)
            
            # Extract validation metrics from trainer
            if trainer.callback_metrics:
                val_metrics = {}
                for key, value in trainer.callback_metrics.items():
                    if isinstance(value, torch.Tensor):
                        val_metrics[key] = value.item()
                    else:
                        val_metrics[key] = value
                
                fold_result = {
                    'fold': fold_idx + 1,
                    'metrics': val_metrics,
                    'model_path': None,
                    'trainer_log_dir': trainer.logger.log_dir if trainer.logger else None
                }
                
                # Save model checkpoint for this fold
                if save_fold_results and trainer.logger:
                    checkpoint_dir = os.path.join(trainer.logger.log_dir, "checkpoints")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    model_path = os.path.join(checkpoint_dir, f"fold_{fold_idx + 1}_final.ckpt")
                    trainer.save_checkpoint(model_path)
                    fold_result['model_path'] = model_path
                
                fold_results.append(fold_result)
                
                print(f"Fold {fold_idx + 1} completed. Metrics: {val_metrics}")
        
        # Restore original config
        self.config['experiment']['name'] = original_exp_name
        self.config['logging']['log_dir'] = original_log_dir
        
        # Aggregate results
        aggregated_results = self._aggregate_fold_results(fold_results)
        
        # Final test evaluation if test set is available
        test_results = None
        if test_loader is not None:
            print(f"\n{'='*50}")
            print("FINAL TEST EVALUATION")
            print(f"{'='*50}")
            test_results = self._evaluate_on_test_set(fold_models, test_loader)
        
        # Prepare final results dictionary
        kfold_results = {
            'n_folds': n_folds,
            'fold_results': fold_results,
            'aggregated_metrics': aggregated_results,
            'test_results': test_results,
            'best_fold': self._find_best_fold(fold_results),
            'fold_models': fold_models if not save_fold_results else None,  # Don't store models if saved to disk
            'config': self.config.copy()
        }
        
        # Save k-fold results
        if save_fold_results:
            results_path = os.path.join(original_log_dir, "kfold_results.pkl")
            os.makedirs(os.path.dirname(results_path), exist_ok=True)
            with open(results_path, 'wb') as f:
                pickle.dump(kfold_results, f)
            print(f"K-fold results saved to: {results_path}")
        
        # Print summary
        self._print_kfold_summary(kfold_results)
        
        return kfold_results
    
    def _aggregate_fold_results(self, fold_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Aggregate metrics across all folds."""
        if not fold_results:
            return {}
        
        # Collect all metric names
        all_metrics = set()
        for fold_result in fold_results:
            if 'metrics' in fold_result:
                all_metrics.update(fold_result['metrics'].keys())
        
        aggregated = {}
        for metric_name in all_metrics:
            values = []
            for fold_result in fold_results:
                if 'metrics' in fold_result and metric_name in fold_result['metrics']:
                    values.append(fold_result['metrics'][metric_name])
            
            if values:
                aggregated[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'values': values
                }
        
        return aggregated
    
    def _evaluate_on_test_set(self, fold_models: List[Any], test_loader: DataLoader) -> Dict[str, Any]:
        """Evaluate all fold models on the test set."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        test_results = {
            'ensemble_predictions': [],
            'individual_predictions': [],
            'true_labels': [],
            'ensemble_metrics': {},
            'individual_metrics': []
        }
        
        # Collect predictions from all models
        all_predictions = []
        true_labels = []
        
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
            
            # Store individual predictions and ensemble
            all_predictions.append(torch.stack(batch_predictions))  # Shape: [n_folds, batch_size, n_tasks]
            true_labels.append(batch.y.cpu())
        
        # Concatenate all batches
        all_predictions = torch.cat(all_predictions, dim=1)  # Shape: [n_folds, total_samples, n_tasks]
        true_labels = torch.cat(true_labels, dim=0)  # Shape: [total_samples, n_tasks]
        
        # Calculate ensemble predictions (mean across folds)
        ensemble_pred = torch.mean(all_predictions, dim=0)  # Shape: [total_samples, n_tasks]
        
        # Calculate metrics
        task_type = self.config['model']['task']
        if task_type == 'regression':
            # Regression metrics
            ensemble_mse = torch.nn.functional.mse_loss(ensemble_pred, true_labels).item()
            ensemble_mae = torch.nn.functional.l1_loss(ensemble_pred, true_labels).item()
            test_results['ensemble_metrics'] = {
                'mse': ensemble_mse,
                'mae': ensemble_mae,
                'rmse': np.sqrt(ensemble_mse)
            }
        elif task_type == 'classification':
            # Classification metrics
            ensemble_pred_class = torch.argmax(ensemble_pred, dim=1)
            true_labels_class = torch.argmax(true_labels, dim=1) if true_labels.dim() > 1 else true_labels.long()
            accuracy = (ensemble_pred_class == true_labels_class).float().mean().item()
            test_results['ensemble_metrics'] = {
                'accuracy': accuracy
            }
        
        # Store predictions and labels
        test_results['ensemble_predictions'] = ensemble_pred.numpy()
        test_results['individual_predictions'] = all_predictions.numpy()
        test_results['true_labels'] = true_labels.numpy()
        
        # Calculate individual model metrics
        for fold_idx in range(len(fold_models)):
            fold_pred = all_predictions[fold_idx]
            if task_type == 'regression':
                fold_mse = torch.nn.functional.mse_loss(fold_pred, true_labels).item()
                fold_mae = torch.nn.functional.l1_loss(fold_pred, true_labels).item()
                test_results['individual_metrics'].append({
                    'fold': fold_idx + 1,
                    'mse': fold_mse,
                    'mae': fold_mae,
                    'rmse': np.sqrt(fold_mse)
                })
            elif task_type == 'classification':
                fold_pred_class = torch.argmax(fold_pred, dim=1)
                true_labels_class = torch.argmax(true_labels, dim=1) if true_labels.dim() > 1 else true_labels.long()
                fold_accuracy = (fold_pred_class == true_labels_class).float().mean().item()
                test_results['individual_metrics'].append({
                    'fold': fold_idx + 1,
                    'accuracy': fold_accuracy
                })
        
        return test_results
    
    def _find_best_fold(self, fold_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find the best performing fold based on the monitoring metric."""
        if not fold_results:
            return None
        
        monitor_metric = self.config['logging']['monitor']
        monitor_mode = self.config['logging']['mode']
        
        best_fold = None
        best_value = None
        
        for fold_result in fold_results:
            if 'metrics' in fold_result and monitor_metric in fold_result['metrics']:
                value = fold_result['metrics'][monitor_metric]
                
                if best_value is None:
                    best_value = value
                    best_fold = fold_result
                elif (monitor_mode == 'min' and value < best_value) or (monitor_mode == 'max' and value > best_value):
                    best_value = value
                    best_fold = fold_result
        
        return best_fold
    
    def _print_kfold_summary(self, kfold_results: Dict[str, Any]) -> None:
        """Print a summary of k-fold cross-validation results."""
        print(f"\n{'='*60}")
        print("K-FOLD CROSS-VALIDATION SUMMARY")
        print(f"{'='*60}")
        
        n_folds = kfold_results['n_folds']
        print(f"Number of folds: {n_folds}")
        
        # Print aggregated metrics
        aggregated = kfold_results['aggregated_metrics']
        if aggregated:
            print(f"\nCross-validation metrics (mean ± std):")
            for metric_name, stats in aggregated.items():
                print(f"  {metric_name}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                      f"(min: {stats['min']:.4f}, max: {stats['max']:.4f})")
        
        # Print best fold
        best_fold = kfold_results['best_fold']
        if best_fold:
            print(f"\nBest fold: {best_fold['fold']}")
            print(f"Best fold metrics:")
            for metric_name, value in best_fold['metrics'].items():
                print(f"  {metric_name}: {value:.4f}")
        
        # Print test results if available
        test_results = kfold_results['test_results']
        if test_results:
            print(f"\nTest set evaluation (ensemble):")
            for metric_name, value in test_results['ensemble_metrics'].items():
                print(f"  {metric_name}: {value:.4f}")
            
            print(f"\nTest set evaluation (individual folds):")
            for fold_metrics in test_results['individual_metrics']:
                fold_num = fold_metrics['fold']
                metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in fold_metrics.items() if k != 'fold'])
                print(f"  Fold {fold_num}: {metrics_str}")
        
        print(f"\n{'='*60}")
    
    def reset(self) -> None:
        """Reset the training manager state."""
        self.trainer = None 