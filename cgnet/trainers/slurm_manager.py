#!/usr/bin/env python
"""SLURM job management for CG-NET training pipeline."""

import os
import yaml
import warnings
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

try:
    import submitit
    SUBMITIT_AVAILABLE = True
    # Suppress submitit warnings about sacct when accounting is disabled
    warnings.filterwarnings("ignore", message=".*sacct error.*", category=UserWarning)
    # Reduce submitit logging verbosity
    logging.getLogger("submitit").setLevel(logging.ERROR)
except ImportError:
    SUBMITIT_AVAILABLE = False


class SlurmManager:
    """Manages SLURM job submission for CG-NET training."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize SlurmManager with configuration.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary
        """
        self.config = config
        if not self._is_slurm_enabled():
            raise RuntimeError(
                "SLURM is not enabled in configuration. Set slurm.use_slurm to true."
            )
        self._check_submitit_available()
    
    def _is_slurm_enabled(self) -> bool:
        """Check if SLURM is enabled in configuration."""
        return self.config.get('slurm', {}).get('use_slurm', False)
    
    def _check_submitit_available(self) -> None:
        """Check if submitit is available."""
        if not SUBMITIT_AVAILABLE:
            raise ImportError(
                "submitit is not installed. Install it with: pip install submitit"
            )
    
    def setup_slurm_executor(self) -> Any:
        """
        Setup submitit SLURM executor with configuration.
            
        Returns
        -------
        submitit.SlurmExecutor
            Configured SLURM executor
        """
        slurm_config = self.config['slurm'].copy()
        
        # Create output directory
        output_path = slurm_config.get('output', 'slurm_logs/cgnet_%j.out')
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup executor
        executor = submitit.AutoExecutor(folder=str(output_dir))
        
        # Configure SLURM parameters
        submitit_params = self._build_submitit_params(slurm_config)
        
        if submitit_params:
            executor.update_parameters(**submitit_params)
        
        return executor
    
    def _build_submitit_params(self, slurm_config: Dict[str, Any]) -> Dict[str, Any]:
        """Build submitit parameters from SLURM config."""
        params = {}
        
        # Basic parameters
        if slurm_config.get('partition'):
            params['slurm_partition'] = slurm_config['partition']
            
        if slurm_config.get('nodes'):
            params['nodes'] = slurm_config['nodes']
        if slurm_config.get('ntasks_per_node'):
            params['ntasks_per_node'] = slurm_config['ntasks_per_node']
        if slurm_config.get('cpus_per_task'):
            params['cpus_per_task'] = slurm_config['cpus_per_task']
        
        # Memory
        if slurm_config.get('mem_gb'):
            params['mem_gb'] = slurm_config['mem_gb']
        elif slurm_config.get('mem'):
            mem_str = slurm_config['mem']
            if isinstance(mem_str, str) and mem_str.upper().endswith('GB'):
                try:
                    params['mem_gb'] = int(mem_str[:-2])
                except ValueError:
                    params['slurm_mem'] = mem_str
            else:
                params['slurm_mem'] = mem_str
        
        # GPU parameters
        if slurm_config.get('gpus_per_node'):
            params['gpus_per_node'] = slurm_config['gpus_per_node']
        elif slurm_config.get('gres'):
            gres_str = slurm_config['gres']
            if isinstance(gres_str, str) and 'gpu:' in gres_str:
                try:
                    gpu_count = int(gres_str.split(':')[-1])
                    params['gpus_per_node'] = gpu_count
                except (ValueError, IndexError):
                    params['slurm_gres'] = gres_str
            else:
                params['slurm_gres'] = gres_str
        
        # Time limit
        if slurm_config.get('time'):
            params['timeout_min'] = self._parse_time_to_minutes(slurm_config['time'])
        
        # Job name
        if slurm_config.get('job_name'):
            params['name'] = slurm_config['job_name']
            
        # Other parameters
        if slurm_config.get('account'):
            params['slurm_account'] = slurm_config['account']
        if slurm_config.get('qos'):
            params['slurm_qos'] = slurm_config['qos']
        
        # Set stderr to stdout by default
        params['stderr_to_stdout'] = True
        
        return params
    
    def _parse_time_to_minutes(self, time_input) -> int:
        """Parse SLURM time input to minutes."""
        # Handle integer input (already in minutes)
        if isinstance(time_input, int):
            return time_input
        
        # Handle string input
        time_str = str(time_input)
        if ':' in time_str:
            parts = time_str.split(':')
            if len(parts) == 3:  # HH:MM:SS
                hours, minutes, seconds = map(int, parts)
                return hours * 60 + minutes + (seconds // 60)
            elif len(parts) == 2:  # MM:SS
                minutes, seconds = map(int, parts)
                return minutes + (seconds // 60)
        else:
            try:
                return int(time_str)
            except ValueError:
                print(f"Warning: Could not parse time '{time_str}', using 60 minutes")
                return 60
    
    def submit_job(self, mode: str = "all", model_path: Optional[str] = None, 
                   config_path: str = None) -> Any:
        """
        Submit training job to SLURM cluster.
        
        Parameters
        ----------
        mode : str
            Mode to run: "data", "train", "test", "predict", or "all"
        model_path : str, optional
            Path to model checkpoint (required for predict mode)
        config_path : str, optional
            Path to configuration file
            
        Returns
        -------
        submitit.Job
            Submitted job object
        """
        # Setup executor
        executor = self.setup_slurm_executor()
        
        # Get config path
        if config_path is None:
            config_path = 'config.yml'
            if not os.path.exists(config_path):
                raise ValueError("No config file found. Please specify config_path.")
        
        config_path = os.path.abspath(config_path)
        
        # Submit job
        job = executor.submit(self._run_cgnet_pipeline, config_path, mode, model_path)
        
        print(f"Job submitted: ID {job.job_id}")
        print(f"Logs: {job.paths.stdout}")
        
        return job
    

    
    @staticmethod
    def _run_cgnet_pipeline(config_path: str, mode: str, model_path: Optional[str] = None):
        """Standalone function to run CG-NET pipeline on SLURM node."""
        import sys
        from pathlib import Path
        
        # Setup environment
        current_dir = Path.cwd()
        project_root = current_dir
        
        # Find project root by looking for cgnet package
        for parent in [current_dir] + list(current_dir.parents):
            if (parent / "cgnet" / "__init__.py").exists():
                project_root = parent
                break
        
        sys.path.insert(0, str(project_root))
        
        try:
            from cgnet.trainers import CGNETTrainer
        except ImportError:
            # Fallback import
            sys.path.insert(0, str(project_root / "cgnet"))
            from trainers import CGNETTrainer
        
        # Run pipeline
        trainer = CGNETTrainer(config_path)
        trainer.run_pipeline(mode, model_path)
        
        return f"Pipeline completed: {mode}"
    
    @staticmethod
    def monitor_jobs(jobs: List[Any]) -> Dict[str, Any]:
        """Monitor status of submitted jobs."""
        if not SUBMITIT_AVAILABLE:
            raise ImportError("submitit is not installed")
        
        status_counts = {}
        job_details = []
        
        for job in jobs:
            state = job.state
            status_counts[state] = status_counts.get(state, 0) + 1
            
            job_info = {
                'job_id': job.job_id,
                'state': state,
                'stdout': str(job.paths.stdout),
            }
            
            if state == 'COMPLETED':
                try:
                    job_info['result'] = job.result()
                except Exception as e:
                    job_info['result'] = f'Error: {e}'
            
            job_details.append(job_info)
        
        return {
            'status_counts': status_counts,
            'total_jobs': len(jobs),
            'job_details': job_details
        }
    
    @staticmethod
    def cancel_jobs(jobs: List[Any]) -> None:
        """Cancel submitted jobs."""
        if not SUBMITIT_AVAILABLE:
            raise ImportError("submitit is not installed")
        
        for job in jobs:
            try:
                job.cancel()
                print(f"Cancelled job {job.job_id}")
            except Exception as e:
                print(f"Failed to cancel job {job.job_id}: {e}") 