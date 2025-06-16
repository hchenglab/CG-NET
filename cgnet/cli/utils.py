#!/usr/bin/env python
"""CLI utilities and helper functions for CG-NET command line interface."""

import os
import argparse
import yaml
import copy
import sys
from typing import Dict, Any, Optional, List
from pathlib import Path

from ..trainers import ConfigManager
from .templates import ConfigTemplateManager
from .validators import ConfigValidator


class CLIUtils:
    """Utility functions for CLI parameter processing and configuration management."""
    
    @staticmethod
    def add_extended_arguments(parser: argparse.ArgumentParser) -> None:
        """Add extended CLI arguments to existing parser."""
        
        template_group = parser.add_argument_group('Template Generation')
        template_group.add_argument(
            "--create-template", type=str, dest="create_template",
            choices=["regression", "classification", "prediction"],
            help="Create configuration template for specific use case"
        )
        template_group.add_argument(
            "--template-output", type=str, dest="template_output",
            help="Output path for generated template (default: print to stdout)"
        )
        template_group.add_argument(
            "--list-templates", action="store_true",
            help="List all available configuration templates"
        )
        
        analysis_group = parser.add_argument_group('Configuration Analysis')
        analysis_group.add_argument(
            "--validate-config", action="store_true",
            help="Validate configuration and show warnings/suggestions"
        )
        analysis_group.add_argument(
            "--suggest-optimizations", action="store_true",
            help="Show optimization suggestions for current configuration"
        )
    
    @staticmethod
    def handle_extended_commands(args: argparse.Namespace, config: Dict[str, Any] = None) -> bool:
        """
        Handle extended CLI commands. 
        
        Parameters
        ----------
        args : argparse.Namespace
            Parsed command line arguments
        config : Dict[str, Any], optional
            Configuration dictionary (required for validation commands)
            
        Returns
        -------
        bool
            True if command was executed and program should exit
        """
        
        # Template generation (doesn't require config)
        if hasattr(args, 'list_templates') and args.list_templates:
            CLIUtils._list_templates()
            return True
        
        if hasattr(args, 'create_template') and args.create_template:
            output = ConfigTemplateManager.create_template(
                args.create_template, 
                args.template_output if hasattr(args, 'template_output') else None
            )
            if args.template_output:
                print(f"✓ Template saved to: {output}")
            else:
                print(output)
            return True
        
        # Commands that require config
        if config is not None:
            if hasattr(args, 'validate_config') and args.validate_config:
                CLIUtils._validate_config_with_output(config)
                return False  # Don't exit, continue with normal flow
            
            if hasattr(args, 'suggest_optimizations') and args.suggest_optimizations:
                CLIUtils._show_optimization_suggestions(config)
                return False  # Don't exit, continue with normal flow
        
        return False
    
    @staticmethod
    def _list_templates() -> None:
        """List all available configuration templates."""
        print("Available Configuration Templates:")
        print("=" * 40)
        templates = ConfigTemplateManager.list_available_templates()
        for template_type, description in templates.items():
            print(f"  {template_type:15} - {description}")
        print("\nUsage: python -m cgnet.cli --create-template <type> [--template-output <path>]")
    
    @staticmethod
    def _validate_config_with_output(config: Dict[str, Any]) -> None:
        """Validate configuration and display results."""
        warnings_list, errors = ConfigValidator.validate_config(config)
        
        if errors:
            print("Configuration Errors:")
            print("-" * 30)
            for error in errors:
                print(f"❌ {error}")
            print()
        
        if warnings_list:
            print("Configuration Warnings:")
            print("-" * 30)
            for warning in warnings_list:
                print(f"⚠ {warning}")
            print()
        
        if not errors and not warnings_list:
            print("✓ Configuration validation passed with no issues")
    
    @staticmethod
    def _show_optimization_suggestions(config: Dict[str, Any]) -> None:
        """Show optimization suggestions for configuration."""
        suggestions = ConfigValidator.suggest_optimizations(config)
        
        if suggestions:
            print("Optimization Suggestions:")
            print("-" * 30)
            for suggestion in suggestions:
                print(f"💡 {suggestion}")
            print()
        else:
            print("✓ No optimization suggestions at this time")
    
    @staticmethod
    def build_config_overrides(args: argparse.Namespace) -> Dict[str, Any]:
        """
        Build configuration override dictionary from CLI arguments.
        
        Parameters
        ----------
        args : argparse.Namespace
            Parsed command line arguments
            
        Returns
        -------
        Dict[str, Any]
            Dictionary of configuration overrides
        """
        overrides = {}
        
        # Experiment section
        if hasattr(args, 'experiment_name') and args.experiment_name is not None:
            overrides.setdefault('experiment', {})['name'] = args.experiment_name
        if hasattr(args, 'seed') and args.seed is not None:
            overrides.setdefault('experiment', {})['seed'] = args.seed
        
        # Data section
        data_overrides = {}
        if hasattr(args, 'data_path') and args.data_path is not None:
            data_overrides['path'] = args.data_path
        if hasattr(args, 'train_ratio') and args.train_ratio is not None:
            data_overrides['train_ratio'] = args.train_ratio
        if hasattr(args, 'val_ratio') and args.val_ratio is not None:
            data_overrides['val_ratio'] = args.val_ratio
        if hasattr(args, 'test_ratio') and args.test_ratio is not None:
            data_overrides['test_ratio'] = args.test_ratio
        if hasattr(args, 'clean_cache') and args.clean_cache:
            data_overrides['clean_cache'] = True
        if hasattr(args, 'no_cache') and args.no_cache:
            data_overrides['save_cache'] = False
        if data_overrides:
            overrides['data'] = data_overrides
        
        # Model section
        model_overrides = {}
        if hasattr(args, 'task') and args.task is not None:
            model_overrides['task'] = args.task
        if hasattr(args, 'hidden_node_dim') and args.hidden_node_dim is not None:
            model_overrides['hidden_node_dim'] = args.hidden_node_dim
        if hasattr(args, 'num_conv') and args.num_conv is not None:
            model_overrides['num_conv'] = args.num_conv
        if hasattr(args, 'n_classes') and args.n_classes is not None:
            model_overrides['n_classes'] = args.n_classes
        if model_overrides:
            overrides['model'] = model_overrides
        
        # Training section
        training_overrides = {}
        if hasattr(args, 'epochs') and args.epochs is not None:
            training_overrides['epochs'] = args.epochs
        if hasattr(args, 'batch_size') and args.batch_size is not None:
            training_overrides['batch_size'] = args.batch_size
        if hasattr(args, 'lr') and args.lr is not None:
            training_overrides['lr'] = args.lr
        if hasattr(args, 'devices') and args.devices is not None:
            training_overrides['devices'] = args.devices
        if hasattr(args, 'num_workers') and args.num_workers is not None:
            training_overrides['num_workers'] = args.num_workers
        
        # Early stopping
        if ((hasattr(args, 'early_stopping') and args.early_stopping) or 
            (hasattr(args, 'early_stopping_patience') and args.early_stopping_patience is not None)):
            early_stop_overrides = {}
            if hasattr(args, 'early_stopping') and args.early_stopping:
                early_stop_overrides['enabled'] = True
            if hasattr(args, 'early_stopping_patience') and args.early_stopping_patience is not None:
                early_stop_overrides['patience'] = args.early_stopping_patience
            training_overrides['early_stopping'] = early_stop_overrides
        
        if training_overrides:
            overrides['training'] = training_overrides
        
        # Cross-validation section
        cv_overrides = {}
        if hasattr(args, 'enable_cv') and args.enable_cv:
            cv_overrides['enabled'] = True
        if hasattr(args, 'n_folds') and args.n_folds is not None:
            cv_overrides['n_folds'] = args.n_folds
        if hasattr(args, 'stratified') and args.stratified:
            cv_overrides['stratified'] = True
        if cv_overrides:
            overrides['cross_validation'] = cv_overrides
        
        # Featurizer section
        feat_overrides = {}
        if hasattr(args, 'featurizer_method') and args.featurizer_method is not None:
            feat_overrides['method'] = args.featurizer_method
        if hasattr(args, 'neighbor_radius') and args.neighbor_radius is not None:
            feat_overrides['neighbor_radius'] = args.neighbor_radius
        if hasattr(args, 'cluster_radius') and args.cluster_radius is not None:
            feat_overrides['cluster_radius'] = args.cluster_radius
        if hasattr(args, 'max_neighbors') and args.max_neighbors is not None:
            feat_overrides['max_neighbors'] = args.max_neighbors
        if feat_overrides:
            overrides['featurizer'] = feat_overrides
        
        # Logging section
        log_overrides = {}
        if hasattr(args, 'log_dir') and args.log_dir is not None:
            log_overrides['log_dir'] = args.log_dir
        if hasattr(args, 'save_top_k') and args.save_top_k is not None:
            log_overrides['save_top_k'] = args.save_top_k
        if log_overrides:
            overrides['logging'] = log_overrides
        
        # SLURM section
        if hasattr(args, 'slurm') and args.slurm:
            slurm_overrides = {'use_slurm': True}
            if hasattr(args, 'partition') and args.partition is not None:
                slurm_overrides['partition'] = args.partition
            if hasattr(args, 'nodes') and args.nodes is not None:
                slurm_overrides['nodes'] = args.nodes
            if hasattr(args, 'cpus_per_task') and args.cpus_per_task is not None:
                slurm_overrides['cpus_per_task'] = args.cpus_per_task
            if hasattr(args, 'mem') and args.mem is not None:
                slurm_overrides['mem'] = args.mem
            if hasattr(args, 'gres') and args.gres is not None:
                slurm_overrides['gres'] = args.gres
            if hasattr(args, 'time') and args.time is not None:
                slurm_overrides['time'] = args.time
            if hasattr(args, 'job_name') and args.job_name is not None:
                slurm_overrides['job_name'] = args.job_name
            overrides['slurm'] = slurm_overrides
        
        # Prediction section
        if hasattr(args, 'output_file') and args.output_file is not None:
            overrides.setdefault('prediction', {})['output_file'] = args.output_file
        
        return overrides
    
    @staticmethod
    def validate_args_consistency(args: argparse.Namespace) -> None:
        """
        Validate argument consistency and combinations.
        
        Parameters
        ----------
        args : argparse.Namespace
            Parsed command line arguments
            
        Raises
        ------
        ValueError
            If arguments are inconsistent or invalid
        """
        # Check config/defaults requirement (skip for template generation)
        if (not hasattr(args, 'config') or not args.config) and \
           (not hasattr(args, 'use_defaults') or not args.use_defaults) and \
           not (hasattr(args, 'create_template') and args.create_template) and \
           not (hasattr(args, 'list_templates') and args.list_templates):
            raise ValueError(
                "Either --config <path> or --use-defaults must be specified"
            )
        
        # Check prediction mode requirements
        if hasattr(args, 'mode') and args.mode == "predict" and \
           (not hasattr(args, 'checkpoint') or not args.checkpoint):
            raise ValueError("Checkpoint path (--checkpoint) is required for prediction mode")
        
        # Check data ratio consistency
        ratios = []
        if hasattr(args, 'train_ratio') and args.train_ratio is not None:
            ratios.append(args.train_ratio)
        if hasattr(args, 'val_ratio') and args.val_ratio is not None:
            ratios.append(args.val_ratio)
        if hasattr(args, 'test_ratio') and args.test_ratio is not None:
            ratios.append(args.test_ratio)
        
        if len(ratios) == 3:
            total = sum(ratios)
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"Data ratios must sum to 1.0, got {total}")
        
        # Check SLURM-specific requirements
        if hasattr(args, 'slurm') and args.slurm:
            if hasattr(args, 'mode') and args.mode == "predict" and \
               (not hasattr(args, 'checkpoint') or not args.checkpoint):
                raise ValueError("Checkpoint path is required for prediction mode with SLURM")
    
    @staticmethod
    def print_config_summary(config: Dict[str, Any], overrides: Dict[str, Any]) -> None:
        """
        Print a summary of the configuration and any CLI overrides.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Final configuration dictionary
        overrides : Dict[str, Any]
            CLI overrides applied to configuration
        """
        print("=" * 60)
        print("CONFIGURATION SUMMARY")
        print("=" * 60)
        
        # Print key configuration details
        print(f"Experiment: {config['experiment']['name']}")
        print(f"Task: {config['model']['task']}")
        print(f"Data path: {config['data']['path']}")
        print(f"Epochs: {config['training']['epochs']}")
        print(f"Batch size: {config['training']['batch_size']}")
        print(f"Learning rate: {config['training']['lr']}")
        
        if config.get('cross_validation', {}).get('enabled', False):
            print(f"Cross-validation: {config['cross_validation']['n_folds']} folds")
        
        if config.get('slurm', {}).get('use_slurm', False):
            print(f"SLURM: Enabled (partition: {config['slurm']['partition']})")
        
        # Print CLI overrides if any
        if overrides:
            print("\nCLI Overrides Applied:")
            print("-" * 30)
            CLIUtils._print_nested_dict(overrides, indent=2)
        
        print("=" * 60)
    
    @staticmethod
    def _print_nested_dict(d: Dict[str, Any], indent: int = 0) -> None:
        """Helper function to print nested dictionary."""
        for key, value in d.items():
            if isinstance(value, dict):
                print(" " * indent + f"{key}:")
                CLIUtils._print_nested_dict(value, indent + 2)
            else:
                print(" " * indent + f"{key}: {value}") 