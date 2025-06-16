#!/usr/bin/env python
"""Main CLI entry point for CG-NET training pipeline."""

import os
import argparse
import yaml
import copy
import sys
import tempfile
from typing import Dict, Any, Optional, Union, List
from pathlib import Path

from ..trainers import ConfigManager, CGNETTrainer
from .utils import CLIUtils


class CLIParameterProcessor:
    """Handles CLI parameter parsing and configuration override logic."""
    
    @staticmethod
    def create_parser() -> argparse.ArgumentParser:
        """Create comprehensive argument parser with all common parameters."""
        parser = argparse.ArgumentParser(
            description="CG-NET Training Pipeline",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Basic usage with config file
  python -m cgnet.cli --config config.yml
  
  # Override specific parameters
  python -m cgnet.cli --config config.yml --batch-size 128 --lr 0.01
  
  # Train with custom experiment name and epochs
  python -m cgnet.cli --config config.yml --exp-name "my_experiment" --epochs 500
  
  # Prediction mode with model checkpoint
  python -m cgnet.cli --config config.yml --mode predict --checkpoint model.ckpt
  
  # Submit to SLURM cluster
  python -m cgnet.cli --config config.yml --slurm
  
  # Use default config (creates config.yml if not exists)
  python -m cgnet.cli --use-defaults
  
  # Create configuration templates
  python -m cgnet.cli --create-template regression --template-output regression_config.yml
  python -m cgnet.cli --list-templates
            """)

        # Required/Core arguments
        core_group = parser.add_argument_group('Core Arguments')
        core_group.add_argument(
            "--config", type=str, 
            help="Path to configuration YAML file (required unless --use-defaults is used)"
        )
        core_group.add_argument(
            "--use-defaults", action="store_true",
            help="Use default configuration (creates config.yml if not exists)"
        )
        core_group.add_argument(
            "--mode", type=str, default=None,
            choices=["all", "train", "predict", "data"],
            help="Training mode: all (default), train, predict, or data (generate dataset only)"
        )

        # Experiment parameters
        exp_group = parser.add_argument_group('Experiment Parameters')
        exp_group.add_argument(
            "--exp-name", type=str, dest="experiment_name",
            help="Override experiment name"
        )
        exp_group.add_argument(
            "--seed", type=int,
            help="Override random seed for reproducibility"
        )

        # Data parameters
        data_group = parser.add_argument_group('Data Parameters')
        data_group.add_argument(
            "--data-path", type=str, dest="data_path",
            help="Override path to raw dataset directory"
        )
        data_group.add_argument(
            "--train-ratio", type=float, dest="train_ratio",
            help="Override fraction of data used for training (0.0-1.0)"
        )
        data_group.add_argument(
            "--val-ratio", type=float, dest="val_ratio",
            help="Override fraction of data used for validation (0.0-1.0)"
        )
        data_group.add_argument(
            "--test-ratio", type=float, dest="test_ratio",
            help="Override fraction of data used for testing (0.0-1.0)"
        )
        data_group.add_argument(
            "--clean-cache", action="store_true",
            help="Clean existing cache and reprocess data"
        )
        data_group.add_argument(
            "--no-cache", action="store_true",
            help="Disable data caching"
        )

        # Model parameters
        model_group = parser.add_argument_group('Model Parameters')
        model_group.add_argument(
            "--task", type=str, choices=["regression", "classification"],
            help="Override task type"
        )
        model_group.add_argument(
            "--hidden-dim", type=int, dest="hidden_node_dim",
            help="Override hidden node feature dimension"
        )
        model_group.add_argument(
            "--num-conv", type=int, dest="num_conv",
            help="Override number of convolutional layers"
        )
        model_group.add_argument(
            "--n-classes", type=int, dest="n_classes",
            help="Override number of classes (for classification)"
        )

        # Training parameters
        train_group = parser.add_argument_group('Training Parameters')
        train_group.add_argument(
            "--epochs", type=int,
            help="Override maximum number of training epochs"
        )
        train_group.add_argument(
            "--batch-size", type=int, dest="batch_size",
            help="Override mini-batch size"
        )
        train_group.add_argument(
            "--lr", "--learning-rate", type=float, dest="lr",
            help="Override initial learning rate"
        )
        train_group.add_argument(
            "--devices", type=int,
            help="Override number of devices to use for training"
        )
        train_group.add_argument(
            "--num-workers", type=int, dest="num_workers",
            help="Override number of data loading workers"
        )
        train_group.add_argument(
            "--early-stopping", action="store_true",
            help="Enable early stopping"
        )
        train_group.add_argument(
            "--early-stopping-patience", type=int, dest="early_stopping_patience",
            help="Override early stopping patience"
        )

        # Cross-validation parameters
        cv_group = parser.add_argument_group('Cross-Validation Parameters')
        cv_group.add_argument(
            "--cross-validation", action="store_true", dest="enable_cv",
            help="Enable k-fold cross-validation"
        )
        cv_group.add_argument(
            "--n-folds", type=int, dest="n_folds",
            help="Override number of folds for cross-validation"
        )
        cv_group.add_argument(
            "--stratified", action="store_true",
            help="Use stratified k-fold"
        )

        # Featurizer parameters
        feat_group = parser.add_argument_group('Featurizer Parameters')
        feat_group.add_argument(
            "--featurizer-method", type=str, dest="featurizer_method",
            choices=["CR", "nth-NN"],
            help="Override featurizer method"
        )
        feat_group.add_argument(
            "--neighbor-radius", type=float, dest="neighbor_radius",
            help="Override cutoff radius for neighbor search (Angstrom)"
        )
        feat_group.add_argument(
            "--cluster-radius", type=float, dest="cluster_radius",
            help="Override cluster radius for CR method (Angstrom)"
        )
        feat_group.add_argument(
            "--max-neighbors", type=int, dest="max_neighbors",
            help="Override maximum number of neighbors per node"
        )

        # Logging parameters
        log_group = parser.add_argument_group('Logging Parameters')
        log_group.add_argument(
            "--log-dir", type=str, dest="log_dir",
            help="Override directory for training logs and checkpoints"
        )
        log_group.add_argument(
            "--save-top-k", type=int, dest="save_top_k",
            help="Override number of best model checkpoints to save"
        )

        # Prediction parameters
        pred_group = parser.add_argument_group('Prediction Parameters')
        pred_group.add_argument(
            "--checkpoint", type=str,
            help="Path to model checkpoint for prediction"
        )
        pred_group.add_argument(
            "--output-file", type=str, dest="output_file",
            help="Override output file for predictions"
        )

        # SLURM parameters
        slurm_group = parser.add_argument_group('SLURM Parameters')
        slurm_group.add_argument(
            "--slurm", action="store_true",
            help="Submit job to SLURM cluster"
        )
        slurm_group.add_argument(
            "--partition", type=str,
            help="Override SLURM partition name"
        )
        slurm_group.add_argument(
            "--nodes", type=int,
            help="Override number of nodes to request"
        )
        slurm_group.add_argument(
            "--cpus-per-task", type=int, dest="cpus_per_task",
            help="Override number of CPUs per task"
        )
        slurm_group.add_argument(
            "--mem", type=str,
            help="Override memory per node (e.g., '32GB')"
        )
        slurm_group.add_argument(
            "--gres", type=str,
            help="Override GPU resources (e.g., 'gpu:1')"
        )
        slurm_group.add_argument(
            "--time", type=str,
            help="Override maximum runtime (HH:MM:SS)"
        )
        slurm_group.add_argument(
            "--job-name", type=str, dest="job_name",
            help="Override SLURM job name"
        )

        # Utility parameters
        util_group = parser.add_argument_group('Utility Parameters')
        util_group.add_argument(
            "--save-config", type=str,
            help="Save final configuration (after CLI overrides) to specified file"
        )
        util_group.add_argument(
            "--print-config", action="store_true",
            help="Print final configuration and exit"
        )
        util_group.add_argument(
            "--validate-only", action="store_true",
            help="Only validate configuration and exit"
        )
        util_group.add_argument(
            "--verbose", "-v", action="store_true",
            help="Enable verbose output"
        )

        # Add extended functionality from CLI utils
        CLIUtils.add_extended_arguments(parser)
        
        return parser
    
    @staticmethod
    def load_or_create_config(config_path: Optional[str], use_defaults: bool) -> Dict[str, Any]:
        """Load configuration from file or create default configuration."""
        if use_defaults:
            if config_path is None:
                config_path = "config.yml"
            
            # Create default config if it doesn't exist
            if not os.path.exists(config_path):
                print(f"Creating default configuration file: {config_path}")
                default_config = ConfigManager.get_default_config()
                ConfigManager.save_config(default_config, config_path)
                print(f"✓ Default configuration saved to {config_path}")
                print("Please review and modify the configuration as needed.")
                return default_config
            else:
                print(f"Loading existing configuration: {config_path}")
                return ConfigManager.load_config(config_path)
        
        elif config_path:
            return ConfigManager.load_config(config_path)
        
        else:
            raise ValueError(
                "Either --config <path> or --use-defaults must be specified. "
                "Use --help for more information."
            )
    
    @staticmethod
    def apply_config_overrides(config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Apply CLI overrides to configuration."""
        return ConfigManager.update_config(config, overrides)


def parse_args():
    """Parse command line arguments with enhanced parameter support."""
    parser = CLIParameterProcessor.create_parser()
    args = parser.parse_args()
    
    # Validate argument consistency
    try:
        CLIUtils.validate_args_consistency(args)
    except ValueError as e:
        parser.error(str(e))
    
    return args


def main():
    """Enhanced main entry point for the CLI."""
    args = parse_args()
    
    try:
        # Load base configuration
        if args.verbose:
            print("Loading configuration...")
        
        # Handle extended commands that don't need config first
        if CLIUtils.handle_extended_commands(args, None):
            return  # Exit if extended command was handled
        
        config = CLIParameterProcessor.load_or_create_config(args.config, args.use_defaults)
        
        # Build and apply CLI overrides
        overrides = CLIUtils.build_config_overrides(args)
        if overrides:
            if args.verbose:
                print("Applying CLI parameter overrides...")
            config = CLIParameterProcessor.apply_config_overrides(config, overrides)
        
        # Handle mode override from CLI
        if args.mode is not None:
            execution_mode = args.mode
        else:
            execution_mode = "all"  # Default mode
        
        # Handle extended commands that need config
        if CLIUtils.handle_extended_commands(args, config):
            return  # Exit if extended command was handled
        
        # Utility functions
        if args.print_config:
            CLIUtils.print_config_summary(config, overrides)
            print("\nFull Configuration:")
            print(yaml.dump(config, default_flow_style=False, indent=2))
            return
        
        if args.save_config:
            ConfigManager.save_config(config, args.save_config)
            print(f"✓ Configuration saved to: {args.save_config}")
            if args.validate_only:
                return
        
        if args.validate_only:
            print("✓ Configuration validation passed")
            return
        
        # Print configuration summary
        if args.verbose:
            CLIUtils.print_config_summary(config, overrides)
        
        # Create trainer instance with final configuration
        # Save config to temporary file for trainer
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
            temp_config_path = f.name
        
        try:
            trainer = CGNETTrainer(temp_config_path)
            
            # Handle SLURM submission
            if config.get('slurm', {}).get('use_slurm', False):
                print("Submitting job to SLURM cluster...")
                
                # Validate SLURM-specific requirements
                if execution_mode == "predict" and not args.checkpoint:
                    raise ValueError("Checkpoint path is required for prediction mode")
                
                # Submit job to SLURM
                job = trainer.submit_pipeline(mode=execution_mode, model_path=args.checkpoint)
                print(f"✓ Job submitted successfully!")
                return

            # Local execution
            if execution_mode == "data":
                trainer.run_pipeline(mode="data")
            elif execution_mode == "all":
                trainer.run_pipeline(mode="all")
            elif execution_mode == "train":
                trainer.run_pipeline(mode="train")
            elif execution_mode == "predict":
                if not args.checkpoint:
                    raise ValueError("Checkpoint path is required for prediction mode")
                trainer.run_pipeline(mode="predict", model_path=args.checkpoint)
            else:
                raise ValueError(f"Unknown mode: {execution_mode}")
        
        finally:
            # Clean up temporary config file
            try:
                os.unlink(temp_config_path)
            except:
                pass
    
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 