#!/usr/bin/env python
"""Main CLI entry point for CG-NET training pipeline."""

import os
import argparse
import yaml
import sys
import tempfile
from typing import Dict, Any, Optional

from ..trainers import ConfigManager, CGNETTrainer


def list_templates() -> None:
    """List all available configuration templates."""
    print("Available Configuration Templates:")
    print("=" * 40)
    templates = ConfigManager.list_available_templates()
    for template_type, description in templates.items():
        print(f"  {template_type:15} - {description}")
    print("\nUsage: python -m cgnet.cli --create-template <type> [--save-config <path>]")


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
    if getattr(args, 'exp_name', None):
        overrides.setdefault('experiment', {})['name'] = args.exp_name
    if getattr(args, 'seed', None):
        overrides.setdefault('experiment', {})['seed'] = args.seed
    
    # Data section
    data_overrides = {}
    if getattr(args, 'data_path', None):
        data_overrides['path'] = args.data_path
    if getattr(args, 'train_ratio', None):
        data_overrides['train_ratio'] = args.train_ratio
    if getattr(args, 'val_ratio', None):
        data_overrides['val_ratio'] = args.val_ratio
    if getattr(args, 'test_ratio', None):
        data_overrides['test_ratio'] = args.test_ratio
    if data_overrides:
        overrides['data'] = data_overrides
    
    # Model section
    model_overrides = {}
    if getattr(args, 'task', None):
        model_overrides['task'] = args.task
    if getattr(args, 'hidden_dim', None):
        model_overrides['hidden_node_dim'] = args.hidden_dim
    if getattr(args, 'n_classes', None):
        model_overrides['n_classes'] = args.n_classes
    if model_overrides:
        overrides['model'] = model_overrides
    
    # Training section
    training_overrides = {}
    if getattr(args, 'epochs', None):
        training_overrides['epochs'] = args.epochs
    if getattr(args, 'batch_size', None):
        training_overrides['batch_size'] = args.batch_size
    if getattr(args, 'lr', None):
        training_overrides['lr'] = args.lr
    if getattr(args, 'devices', None):
        training_overrides['devices'] = args.devices
    if training_overrides:
        overrides['training'] = training_overrides
    
    # SLURM section
    if getattr(args, 'slurm', False):
        overrides['slurm'] = {'use_slurm': True}
    
    # Prediction section
    prediction_overrides = {}
    if getattr(args, 'output_file', None):
        prediction_overrides['output_file'] = args.output_file
    if getattr(args, 'checkpoint', None):
        prediction_overrides['model_path'] = args.checkpoint
    if getattr(args, 'batch_size', None):
        prediction_overrides['batch_size'] = args.batch_size
    if prediction_overrides:
        overrides['prediction'] = prediction_overrides
    
    return overrides


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser with essential parameters."""
    parser = argparse.ArgumentParser(
        description="CG-NET Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with config file
  python -m cgnet.cli --config config.yml
  
  # Override specific parameters
  python -m cgnet.cli --config config.yml --batch-size 128 --lr 0.01
  
  # Prediction mode with model checkpoint
  python -m cgnet.cli --config config.yml --mode predict --checkpoint model.ckpt
  
  # Use default config
  python -m cgnet.cli --use-defaults
        """)

    # Core arguments
    parser.add_argument(
        "--config", type=str, 
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--use-defaults", action="store_true",
        help="Use default configuration (creates config.yml if not exists)"
    )
    parser.add_argument(
        "--mode", type=str, default="all",
        choices=["all", "train", "predict", "data"],
        help="Training mode: all (default), train, predict, or data"
    )

    # Essential experiment parameters
    parser.add_argument("--exp-name", type=str, help="Experiment name")
    parser.add_argument("--seed", type=int, help="Random seed")

    # Essential data parameters
    parser.add_argument("--data-path", type=str, help="Path to dataset")
    parser.add_argument("--train-ratio", type=float, help="Training data ratio")
    parser.add_argument("--val-ratio", type=float, help="Validation data ratio")
    parser.add_argument("--test-ratio", type=float, help="Test data ratio")

    # Essential model parameters
    parser.add_argument("--task", type=str, choices=["regression", "classification"], help="Task type")
    parser.add_argument("--hidden-dim", type=int, help="Hidden dimension")
    parser.add_argument("--n-classes", type=int, help="Number of classes")

    # Essential training parameters
    parser.add_argument("--epochs", type=int, help="Training epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--devices", type=int, help="Number of devices")

    # Prediction parameters
    parser.add_argument("--checkpoint", type=str, help="Model checkpoint path")
    parser.add_argument("--output-file", type=str, help="Prediction output file")

    # SLURM support
    parser.add_argument("--slurm", action="store_true", help="Submit to SLURM")

    # Utility parameters
    parser.add_argument("--save-config", type=str, help="Save final config to file")
    parser.add_argument("--print-config", action="store_true", help="Print config and exit")
    parser.add_argument("--validate-only", action="store_true", help="Only validate config")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    # Template support
    parser.add_argument("--create-template", type=str, 
                       choices=["regression", "classification", "prediction"],
                       help="Create configuration template")
    parser.add_argument("--list-templates", action="store_true", help="List templates")

    return parser


def load_or_create_config(config_path: Optional[str], use_defaults: bool) -> Dict[str, Any]:
    """Load configuration from file or create default."""
    return ConfigManager.load_or_create_config(config_path, use_defaults)


def apply_config_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Apply CLI overrides to configuration."""
    overrides = build_config_overrides(args)
    if overrides:
        return ConfigManager.update_config(config, overrides)
    return config


def parse_args():
    """Parse command line arguments."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Basic validation
    if (not args.config and not args.use_defaults and 
        not args.create_template and not args.list_templates):
        parser.error("Either --config <path> or --use-defaults must be specified")
    
    if args.mode == "predict" and not args.checkpoint:
        parser.error("Checkpoint path (--checkpoint) is required for prediction mode")
    
    return args


def main():
    """Main entry point for the CLI."""
    args = parse_args()
    
    try:
        # Handle template commands first
        if args.list_templates:
            list_templates()
            return
        
        if args.create_template:
            output = ConfigManager.create_template(args.create_template, args.save_config)
            if args.save_config:
                print(f"Template saved to: {output}")
            else:
                print(output)
            return
        
        # Load configuration
        if args.verbose:
            print("Loading configuration...")
        
        config = load_or_create_config(args.config, args.use_defaults)
        
        # Apply CLI overrides
        config = apply_config_overrides(config, args)
        
        # Handle utility commands
        if args.save_config:
            ConfigManager.save_config(config, args.save_config)
            print(f"Configuration saved to: {args.save_config}")
        
        if args.print_config:
            print("Configuration:")
            print(yaml.dump(config, default_flow_style=False, indent=2))
            return
        
        if args.validate_only:
            return
        
        if args.validate_only:
            warnings, errors = ConfigManager.validate_config(config)
            if errors:
                print("Validation failed with errors:")
                for error in errors:
                    print(f"  - {error}")
                sys.exit(1)
            else:
                print("Configuration validation passed")
            return
        
        # Print summary if verbose
        if args.verbose:
            warnings, errors = ConfigManager.validate_config(config)
            if warnings:
                print("Configuration warnings:")
                for warning in warnings:
                    print(f"  Warning: {warning}")
        
        # Handle SLURM submission (before creating temporary files)
        if config.get('slurm', {}).get('use_slurm', False) or args.slurm:
            # For SLURM submission, use original config file if available
            if args.config and os.path.exists(args.config):
                # Apply overrides to original config file and save to a persistent location
                if args.verbose:
                    print("Preparing configuration for SLURM submission...")
                
                # Create SLURM logs directory if it doesn't exist
                slurm_log_dir = config.get('slurm', {}).get('output', 'slurm_logs').split('%')[0].rstrip('/')
                if not slurm_log_dir:
                    slurm_log_dir = 'slurm_logs'
                
                os.makedirs(slurm_log_dir, exist_ok=True)
                
                # Save config to SLURM logs directory
                slurm_config_path = os.path.join(slurm_log_dir, 'config_for_slurm.yml')
                ConfigManager.save_config(config, slurm_config_path)
                
                if args.verbose:
                    print(f"Configuration saved to: {slurm_config_path}")
                    print("Submitting job to SLURM cluster...")
                
                # Create trainer with the SLURM config
                trainer = CGNETTrainer(slurm_config_path)
                trainer.submit_pipeline(mode=args.mode, model_path=args.checkpoint)
                print("Job submitted successfully!")
                return
            else:
                # No original config file, create a persistent one
                persistent_config_path = 'config_for_slurm.yml'
                ConfigManager.save_config(config, persistent_config_path)
                
                if args.verbose:
                    print(f"Configuration saved to: {persistent_config_path}")
                    print("Submitting job to SLURM cluster...")
                
                trainer = CGNETTrainer(persistent_config_path)
                trainer.submit_pipeline(mode=args.mode, model_path=args.checkpoint)
                print("Job submitted successfully!")
                return

        # Local execution - use temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
            temp_config_path = f.name
        
        try:
            trainer = CGNETTrainer(temp_config_path)
            
            # Local execution
            if args.mode == "predict" and not args.checkpoint:
                raise ValueError("Checkpoint path is required for prediction mode")
            
            trainer.run_pipeline(mode=args.mode, model_path=args.checkpoint)
        
        finally:
            # Clean up temporary config file
            try:
                os.unlink(temp_config_path)
            except OSError:
                pass
    
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 