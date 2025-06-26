#!/usr/bin/env python
"""SLURM job management for CG-NET training pipeline.

This module provides comprehensive SLURM cluster integration capabilities including:
- Job submission and configuration management
- Resource allocation and parameter validation
- Job monitoring and status tracking
- Error handling and dependency management
- Environment setup and path resolution

The SlurmManager class serves as the central interface for all SLURM-related
operations in the CG-NET training pipeline.
"""

import os
import sys
import yaml
import warnings
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from enum import Enum
from dataclasses import dataclass, field
import time

# Configure logging
logger = logging.getLogger(__name__)

# Handle submitit import with graceful degradation
try:
    import submitit
    SUBMITIT_AVAILABLE = True
    
    # Suppress submitit warnings about sacct when accounting is disabled
    warnings.filterwarnings("ignore", message=".*sacct error.*", category=UserWarning)
    
    # Reduce submitit logging verbosity
    logging.getLogger("submitit").setLevel(logging.ERROR)
    
except ImportError:
    SUBMITIT_AVAILABLE = False
    submitit = None


class SlurmError(Exception):
    """Base exception for SLURM-related errors."""
    pass


class SlurmConfigurationError(SlurmError):
    """Exception raised for SLURM configuration errors."""
    pass


class SlurmSubmissionError(SlurmError):
    """Exception raised for job submission errors."""
    pass


class SlurmDependencyError(SlurmError):
    """Exception raised for dependency-related errors."""
    pass


class JobState(Enum):
    """Enumeration of SLURM job states."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMEOUT = "TIMEOUT"
    UNKNOWN = "UNKNOWN"


class PipelineMode(Enum):
    """Enumeration of pipeline execution modes."""
    ALL = "all"
    DATA = "data"
    TRAIN = "train"
    TEST = "test"
    PREDICT = "predict"


@dataclass
class SlurmJobInfo:
    """Data class for SLURM job information."""
    job_id: str
    state: JobState
    stdout_path: Optional[str] = None
    stderr_path: Optional[str] = None
    result: Optional[Any] = None
    error_message: Optional[str] = None
    submission_time: Optional[float] = None
    completion_time: Optional[float] = None
    
    @property
    def is_finished(self) -> bool:
        """Check if job has finished (completed, failed, cancelled, or timeout)."""
        return self.state in [JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED, JobState.TIMEOUT]
    
    @property
    def is_successful(self) -> bool:
        """Check if job completed successfully."""
        return self.state == JobState.COMPLETED
    
    @property
    def runtime(self) -> Optional[float]:
        """Get job runtime in seconds."""
        if self.submission_time and self.completion_time:
            return self.completion_time - self.submission_time
        return None


class SlurmConstants:
    """Constants used throughout SLURM management."""
    
    # Default values
    DEFAULT_OUTPUT_DIR = "slurm_logs"
    DEFAULT_OUTPUT_PATTERN = "cgnet_%j.out"
    DEFAULT_ERROR_PATTERN = "cgnet_%j.err"
    DEFAULT_TIMEOUT_MINUTES = 60
    DEFAULT_NODES = 1
    DEFAULT_NTASKS_PER_NODE = 1
    DEFAULT_CPUS_PER_TASK = 1
    DEFAULT_MEM_GB = 4
    
    # Time format patterns
    TIME_FORMAT_HH_MM_SS = r"^\d{1,2}:\d{2}:\d{2}$"
    TIME_FORMAT_MM_SS = r"^\d{1,2}:\d{2}$"
    TIME_FORMAT_MINUTES = r"^\d+$"
    
    # Memory patterns
    MEMORY_PATTERN_GB = r"(\d+)GB?$"
    MEMORY_PATTERN_MB = r"(\d+)MB?$"
    
    # GPU patterns
    GPU_GRES_PATTERN = r"gpu:(\d+)"
    
    # Required configuration keys
    REQUIRED_CONFIG_KEYS = ["use_slurm"]


class SlurmValidator:
    """Handles SLURM configuration validation."""
    
    @staticmethod
    def validate_slurm_config(config: Dict[str, Any]) -> None:
        """
        Validate SLURM configuration.
        
        Args:
            config: Configuration dictionary to validate
            
        Raises:
            SlurmConfigurationError: If configuration is invalid
        """
        if 'slurm' not in config:
            raise SlurmConfigurationError("Missing 'slurm' section in configuration")
        
        slurm_config = config['slurm']
        
        # Check required keys
        missing_keys = [key for key in SlurmConstants.REQUIRED_CONFIG_KEYS 
                       if key not in slurm_config]
        if missing_keys:
            raise SlurmConfigurationError(f"Missing required SLURM configuration keys: {missing_keys}")
        
        # Validate use_slurm
        if not isinstance(slurm_config.get('use_slurm'), bool):
            raise SlurmConfigurationError("'use_slurm' must be a boolean value")
        
        # Validate numeric parameters
        SlurmValidator._validate_numeric_parameters(slurm_config)
        
        # Validate time format
        if 'time' in slurm_config:
            SlurmValidator._validate_time_format(slurm_config['time'])
        
        # Validate memory format
        if 'mem' in slurm_config or 'mem_gb' in slurm_config:
            SlurmValidator._validate_memory_format(slurm_config)
    
    @staticmethod
    def _validate_numeric_parameters(slurm_config: Dict[str, Any]) -> None:
        """Validate numeric SLURM parameters."""
        numeric_params = {
            'nodes': (1, 1000, "Nodes"),
            'ntasks_per_node': (1, 128, "Tasks per node"),
            'cpus_per_task': (1, 128, "CPUs per task"),
            'mem_gb': (1, 1000, "Memory in GB"),
            'gpus_per_node': (0, 8, "GPUs per node")
        }
        
        for param, (min_val, max_val, desc) in numeric_params.items():
            if param in slurm_config:
                value = slurm_config[param]
                if not isinstance(value, int) or value < min_val or value > max_val:
                    raise SlurmConfigurationError(
                        f"{desc} must be an integer between {min_val} and {max_val}, got {value}"
                    )
    
    @staticmethod
    def _validate_time_format(time_value: Union[str, int]) -> None:
        """Validate time format."""
        if isinstance(time_value, int):
            if time_value <= 0:
                raise SlurmConfigurationError("Time must be positive")
            return
        
        if not isinstance(time_value, str):
            raise SlurmConfigurationError("Time must be string or integer")
        
        # Check common time formats
        import re
        if not (re.match(SlurmConstants.TIME_FORMAT_HH_MM_SS, time_value) or
                re.match(SlurmConstants.TIME_FORMAT_MM_SS, time_value) or
                re.match(SlurmConstants.TIME_FORMAT_MINUTES, time_value)):
            raise SlurmConfigurationError(
                f"Invalid time format: {time_value}. "
                "Use HH:MM:SS, MM:SS, or integer minutes"
            )
    
    @staticmethod
    def _validate_memory_format(slurm_config: Dict[str, Any]) -> None:
        """Validate memory format."""
        if 'mem_gb' in slurm_config:
            mem_gb = slurm_config['mem_gb']
            if not isinstance(mem_gb, (int, float)) or mem_gb <= 0:
                raise SlurmConfigurationError("mem_gb must be a positive number")
        
        if 'mem' in slurm_config:
            mem_str = slurm_config['mem']
            if not isinstance(mem_str, str):
                raise SlurmConfigurationError("mem must be a string")
            
            import re
            if not (re.match(SlurmConstants.MEMORY_PATTERN_GB, mem_str.upper()) or
                    re.match(SlurmConstants.MEMORY_PATTERN_MB, mem_str.upper()) or
                    mem_str.isdigit()):
                raise SlurmConfigurationError(
                    f"Invalid memory format: {mem_str}. Use format like '4GB', '4000MB', or '4000'"
                )


class SlurmParameterBuilder:
    """Builds submitit parameters from SLURM configuration."""
    
    @staticmethod
    def build_submitit_params(slurm_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build submitit parameters from SLURM configuration.
        
        Args:
            slurm_config: SLURM configuration dictionary
            
        Returns:
            Dictionary of submitit parameters
        """
        params = {}
        
        # Basic job parameters
        SlurmParameterBuilder._add_basic_parameters(params, slurm_config)
        
        # Resource parameters
        SlurmParameterBuilder._add_resource_parameters(params, slurm_config)
        
        # Time and scheduling parameters
        SlurmParameterBuilder._add_scheduling_parameters(params, slurm_config)
        
        # Output parameters
        SlurmParameterBuilder._add_output_parameters(params, slurm_config)
        
        return params
    
    @staticmethod
    def _add_basic_parameters(params: Dict[str, Any], slurm_config: Dict[str, Any]) -> None:
        """Add basic job parameters."""
        if slurm_config.get('partition'):
            params['slurm_partition'] = slurm_config['partition']
        
        if slurm_config.get('job_name'):
            params['name'] = slurm_config['job_name']
        
        if slurm_config.get('account'):
            params['slurm_account'] = slurm_config['account']
        
        if slurm_config.get('qos'):
            params['slurm_qos'] = slurm_config['qos']
    
    @staticmethod
    def _add_resource_parameters(params: Dict[str, Any], slurm_config: Dict[str, Any]) -> None:
        """Add resource allocation parameters."""
        # Node and task parameters
        if slurm_config.get('nodes'):
            params['nodes'] = slurm_config['nodes']
        
        if slurm_config.get('ntasks_per_node'):
            params['slurm_ntasks_per_node'] = slurm_config['ntasks_per_node']
        
        if slurm_config.get('cpus_per_task'):
            params['cpus_per_task'] = slurm_config['cpus_per_task']
        
        # Memory parameters
        SlurmParameterBuilder._add_memory_parameters(params, slurm_config)
        
        # GPU parameters
        SlurmParameterBuilder._add_gpu_parameters(params, slurm_config)
    
    @staticmethod
    def _add_memory_parameters(params: Dict[str, Any], slurm_config: Dict[str, Any]) -> None:
        """Add memory parameters."""
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
    
    @staticmethod
    def _add_gpu_parameters(params: Dict[str, Any], slurm_config: Dict[str, Any]) -> None:
        """Add GPU parameters."""
        if slurm_config.get('gpus_per_node'):
            params['gpus_per_node'] = slurm_config['gpus_per_node']
        elif slurm_config.get('gres'):
            gres_str = slurm_config['gres']
            
            if isinstance(gres_str, str) and 'gpu:' in gres_str:
                import re
                match = re.search(SlurmConstants.GPU_GRES_PATTERN, gres_str)
                if match:
                    try:
                        params['gpus_per_node'] = int(match.group(1))
                    except ValueError:
                        params['slurm_gres'] = gres_str
                else:
                    params['slurm_gres'] = gres_str
            else:
                params['slurm_gres'] = gres_str
    
    @staticmethod
    def _add_scheduling_parameters(params: Dict[str, Any], slurm_config: Dict[str, Any]) -> None:
        """Add scheduling and time parameters."""
        if slurm_config.get('time'):
            params['timeout_min'] = SlurmParameterBuilder._parse_time_to_minutes(slurm_config['time'])
    
    @staticmethod
    def _add_output_parameters(params: Dict[str, Any], slurm_config: Dict[str, Any]) -> None:
        """Add output and logging parameters."""
        # Set stderr to stdout by default for simplicity
        params['stderr_to_stdout'] = True
    
    @staticmethod
    def _parse_time_to_minutes(time_input: Union[str, int]) -> int:
        """
        Parse SLURM time input to minutes.
        
        Args:
            time_input: Time specification (string or integer)
            
        Returns:
            Time in minutes
        """
        # Handle integer input (already in minutes)
        if isinstance(time_input, int):
            return max(1, time_input)
        
        # Handle string input
        time_str = str(time_input).strip()
        
        try:
            if ':' in time_str:
                parts = time_str.split(':')
                if len(parts) == 3:  # HH:MM:SS
                    hours, minutes, seconds = map(int, parts)
                    return max(1, hours * 60 + minutes + (seconds + 59) // 60)  # Round up seconds
                elif len(parts) == 2:  # MM:SS
                    minutes, seconds = map(int, parts)
                    return max(1, minutes + (seconds + 59) // 60)  # Round up seconds
            else:
                # Just minutes
                return max(1, int(time_str))
        except (ValueError, IndexError) as e:
            logger.warning(f"Could not parse time '{time_input}', using default {SlurmConstants.DEFAULT_TIMEOUT_MINUTES} minutes: {e}")
            return SlurmConstants.DEFAULT_TIMEOUT_MINUTES


class SlurmJobMonitor:
    """Handles SLURM job monitoring and status tracking."""
    
    @staticmethod
    def get_job_info(job: Any) -> SlurmJobInfo:
        """
        Get comprehensive information about a SLURM job.
        
        Args:
            job: submitit job object
            
        Returns:
            SlurmJobInfo object containing job details
        """
        try:
            state_str = str(job.state)
            state = JobState(state_str) if state_str in [s.value for s in JobState] else JobState.UNKNOWN
        except (AttributeError, ValueError):
            state = JobState.UNKNOWN
        
        job_info = SlurmJobInfo(
            job_id=str(job.job_id) if hasattr(job, 'job_id') else 'unknown',
            state=state,
            submission_time=time.time()  # Approximate submission time
        )
        
        # Add paths if available
        if hasattr(job, 'paths'):
            if hasattr(job.paths, 'stdout'):
                job_info.stdout_path = str(job.paths.stdout)
            if hasattr(job.paths, 'stderr'):
                job_info.stderr_path = str(job.paths.stderr)
        
        # Get result if job is completed
        if job_info.is_finished:
            job_info.completion_time = time.time()
            
            if job_info.is_successful:
                try:
                    job_info.result = job.result()
                except Exception as e:
                    job_info.error_message = f"Error retrieving result: {str(e)}"
                    logger.warning(f"Could not retrieve result for job {job_info.job_id}: {e}")
            elif state == JobState.FAILED:
                try:
                    job_info.error_message = str(job.exception())
                except Exception as e:
                    job_info.error_message = f"Error retrieving exception: {str(e)}"
        
        return job_info
    
    @staticmethod
    def monitor_jobs(jobs: List[Any]) -> Dict[str, Any]:
        """
        Monitor status of multiple SLURM jobs.
        
        Args:
            jobs: List of submitit job objects
            
        Returns:
            Dictionary containing comprehensive job monitoring information
        """
        if not SUBMITIT_AVAILABLE:
            raise SlurmDependencyError("submitit is not installed")
        
        status_counts = {}
        job_details = []
        total_runtime = 0
        completed_jobs = 0
        
        for job in jobs:
            job_info = SlurmJobMonitor.get_job_info(job)
            
            # Update status counts
            state_str = job_info.state.value
            status_counts[state_str] = status_counts.get(state_str, 0) + 1
            
            # Convert to dictionary for JSON serialization
            job_dict = {
                'job_id': job_info.job_id,
                'state': job_info.state.value,
                'stdout_path': job_info.stdout_path,
                'stderr_path': job_info.stderr_path,
                'submission_time': job_info.submission_time,
                'completion_time': job_info.completion_time,
                'runtime': job_info.runtime
            }
            
            if job_info.result is not None:
                job_dict['result'] = job_info.result
            
            if job_info.error_message:
                job_dict['error_message'] = job_info.error_message
            
            job_details.append(job_dict)
            
            # Calculate statistics
            if job_info.runtime:
                total_runtime += job_info.runtime
                completed_jobs += 1
        
        # Calculate summary statistics
        avg_runtime = total_runtime / completed_jobs if completed_jobs > 0 else 0
        success_rate = (status_counts.get(JobState.COMPLETED.value, 0) / len(jobs)) * 100 if jobs else 0
        
        return {
            'status_counts': status_counts,
            'total_jobs': len(jobs),
            'completed_jobs': completed_jobs,
            'success_rate': round(success_rate, 2),
            'average_runtime': round(avg_runtime, 2),
            'job_details': job_details,
            'monitoring_time': time.time()
        }


class SlurmManager:
    """Manages SLURM job submission and monitoring for CG-NET training.
    
    This class provides a comprehensive interface for SLURM cluster integration including:
    - Job submission with parameter validation and configuration
    - Resource allocation and constraint management
    - Job monitoring and status tracking
    - Error handling and dependency validation
    - Environment setup and path resolution
    
    Attributes:
        config (Dict[str, Any]): Configuration dictionary
        _executor (Optional[Any]): Cached submitit executor instance
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize SlurmManager with configuration.
        
        Args:
            config: Configuration dictionary containing SLURM settings
            
        Raises:
            SlurmConfigurationError: If SLURM configuration is invalid
            SlurmDependencyError: If required dependencies are missing
        """
        self.config = config
        self._executor: Optional[Any] = None
        
        # Validate configuration
        SlurmValidator.validate_slurm_config(config)
        
        if not self._is_slurm_enabled():
            raise SlurmConfigurationError(
                "SLURM is not enabled in configuration. Set slurm.use_slurm to true."
            )
        
        self._check_submitit_available()
        
        logger.info("SlurmManager initialized successfully")
    
    def _is_slurm_enabled(self) -> bool:
        """Check if SLURM is enabled in configuration."""
        return self.config.get('slurm', {}).get('use_slurm', False)
    
    def _check_submitit_available(self) -> None:
        """Check if submitit is available."""
        if not SUBMITIT_AVAILABLE:
            raise SlurmDependencyError(
                "submitit is not installed. Install it with: pip install submitit"
            )
    
    def setup_slurm_executor(self, force_recreate: bool = False) -> Any:
        """
        Setup submitit SLURM executor with configuration.
        
        Args:
            force_recreate: Whether to force recreation of executor
            
        Returns:
            Configured submitit executor
            
        Raises:
            SlurmConfigurationError: If executor setup fails
        """
        # Return cached executor if available
        if self._executor is not None and not force_recreate:
            return self._executor
        
        logger.info("Setting up SLURM executor...")
        
        try:
            slurm_config = self.config['slurm'].copy()
            
            # Setup output directory
            output_dir = self._setup_output_directory(slurm_config)
            
            # Create executor
            executor = submitit.AutoExecutor(folder=str(output_dir))
            
            # Configure SLURM parameters
            submitit_params = SlurmParameterBuilder.build_submitit_params(slurm_config)
            
            if submitit_params:
                executor.update_parameters(**submitit_params)
                logger.debug(f"Executor configured with parameters: {submitit_params}")
            
            # Cache executor
            self._executor = executor
            
            logger.info("SLURM executor setup completed successfully")
            return executor
            
        except Exception as e:
            raise SlurmConfigurationError(f"Failed to setup SLURM executor: {str(e)}") from e
    
    def _setup_output_directory(self, slurm_config: Dict[str, Any]) -> Path:
        """Setup and create output directory for SLURM logs."""
        output_path = slurm_config.get('output', 
                                     f"{SlurmConstants.DEFAULT_OUTPUT_DIR}/{SlurmConstants.DEFAULT_OUTPUT_PATTERN}")
        output_dir = Path(output_path).parent
        
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Output directory created: {output_dir}")
            return output_dir
        except Exception as e:
            raise SlurmConfigurationError(f"Failed to create output directory {output_dir}: {str(e)}") from e
    
    def validate_pipeline_mode(self, mode: str) -> PipelineMode:
        """
        Validate and convert pipeline mode string.
        
        Args:
            mode: Mode string to validate
            
        Returns:
            PipelineMode enum value
            
        Raises:
            SlurmConfigurationError: If mode is invalid
        """
        try:
            return PipelineMode(mode)
        except ValueError:
            valid_modes = [m.value for m in PipelineMode]
            raise SlurmConfigurationError(f"Invalid pipeline mode: {mode}. Must be one of: {valid_modes}")
    
    def submit_job(self, 
                   mode: str = "all", 
                   model_path: Optional[str] = None, 
                   config_path: Optional[str] = None) -> Any:
        """
        Submit training job to SLURM cluster.
        
        Args:
            mode: Pipeline mode to run ("data", "train", "test", "predict", or "all")
            model_path: Path to model checkpoint (required for predict mode)
            config_path: Path to configuration file
            
        Returns:
            submitit.Job object representing the submitted job
            
        Raises:
            SlurmSubmissionError: If job submission fails
            SlurmConfigurationError: If configuration is invalid
        """
        logger.info(f"Submitting SLURM job with mode: {mode}")
        
        try:
            # Validate mode
            pipeline_mode = self.validate_pipeline_mode(mode)
            
            # Validate model path for prediction mode
            if pipeline_mode == PipelineMode.PREDICT and not model_path:
                raise SlurmConfigurationError("model_path is required for predict mode")
            
            # Setup executor
            executor = self.setup_slurm_executor()
            
            # Resolve config path
            config_path = self._resolve_config_path(config_path)
            
            # Submit job
            job = executor.submit(self._run_cgnet_pipeline, config_path, mode, model_path)
            
            logger.info(f"Job submitted successfully: ID {job.job_id}")
            logger.info(f"Logs will be written to: {job.paths.stdout}")
            
            return job
            
        except Exception as e:
            raise SlurmSubmissionError(f"Failed to submit job: {str(e)}") from e
    
    def _resolve_config_path(self, config_path: Optional[str]) -> str:
        """Resolve and validate configuration file path."""
        if config_path is None:
            config_path = 'config.yml'
        
        # Convert to absolute path
        config_path = os.path.abspath(config_path)
        
        # Check if the file exists
        if not os.path.exists(config_path):
            raise SlurmConfigurationError(f"Configuration file not found: {config_path}")
        
        # For SLURM execution, ensure the config file will be accessible from compute nodes
        # If it's in /tmp, warn the user as it might not be accessible
        if config_path.startswith('/tmp'):
            logger.warning(f"Configuration file is in /tmp directory: {config_path}")
            logger.warning("This may not be accessible from SLURM compute nodes")
        
        logger.debug(f"Using configuration file: {config_path}")
        return config_path
    
    def submit_multiple_jobs(self, 
                           job_configs: List[Dict[str, Any]]) -> List[Any]:
        """
        Submit multiple jobs with different configurations.
        
        Args:
            job_configs: List of job configuration dictionaries, each containing
                        'mode', optional 'model_path', and optional 'config_path'
            
        Returns:
            List of submitted job objects
            
        Raises:
            SlurmSubmissionError: If any job submission fails
        """
        logger.info(f"Submitting {len(job_configs)} SLURM jobs...")
        
        submitted_jobs = []
        failed_submissions = []
        
        for i, job_config in enumerate(job_configs):
            try:
                job = self.submit_job(
                    mode=job_config.get('mode', 'all'),
                    model_path=job_config.get('model_path'),
                    config_path=job_config.get('config_path')
                )
                submitted_jobs.append(job)
                logger.info(f"Job {i+1}/{len(job_configs)} submitted: {job.job_id}")
                
            except Exception as e:
                error_msg = f"Job {i+1}/{len(job_configs)} failed: {str(e)}"
                logger.error(error_msg)
                failed_submissions.append((i, error_msg))
        
        if failed_submissions:
            error_summary = "; ".join([f"Job {i}: {msg}" for i, msg in failed_submissions])
            logger.warning(f"Some jobs failed to submit: {error_summary}")
        
        logger.info(f"Successfully submitted {len(submitted_jobs)}/{len(job_configs)} jobs")
        return submitted_jobs
    
    @staticmethod
    def _run_cgnet_pipeline(config_path: str, mode: str, model_path: Optional[str] = None) -> str:
        """
        Standalone function to run CG-NET pipeline on SLURM node.
        
        This function is executed on the compute node and handles:
        - Environment setup and path resolution
        - Module imports and dependency validation
        - Pipeline execution and error handling
        
        Args:
            config_path: Path to configuration file
            mode: Pipeline execution mode
            model_path: Optional path to model checkpoint
            
        Returns:
            Success message string
            
        Raises:
            ImportError: If CG-NET modules cannot be imported
            RuntimeError: If pipeline execution fails
        """
        logger.info(f"Starting CG-NET pipeline on SLURM node: mode={mode}, config={config_path}")
        
        try:
            # Check if config file exists before proceeding
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
            logger.info(f"Configuration file found: {config_path}")
            
            # Setup environment and paths
            project_root = SlurmManager._find_project_root()
            SlurmManager._setup_python_path(project_root)
            
            # Import trainer
            trainer_class = SlurmManager._import_trainer()
            
            # Execute pipeline
            logger.info(f"Creating trainer with config: {config_path}")
            trainer = trainer_class(config_path)
            
            logger.info(f"Running pipeline with mode: {mode}")
            result = trainer.run_pipeline(mode, model_path)
            
            success_msg = f"Pipeline completed successfully: {mode}"
            logger.info(success_msg)
            
            return success_msg
            
        except Exception as e:
            error_msg = f"Pipeline execution failed: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Current working directory: {os.getcwd()}")
            logger.error(f"Python path: {sys.path}")
            raise RuntimeError(error_msg) from e
    
    @staticmethod
    def _find_project_root() -> Path:
        """Find the project root directory by looking for cgnet package."""
        current_dir = Path.cwd()
        
        # Search in current directory and parent directories
        search_paths = [current_dir] + list(current_dir.parents)
        
        for parent in search_paths:
            cgnet_init = parent / "cgnet" / "__init__.py"
            if cgnet_init.exists():
                logger.debug(f"Found project root: {parent}")
                return parent
        
        # Fallback to current directory
        logger.warning("Could not find project root, using current directory")
        return current_dir
    
    @staticmethod
    def _setup_python_path(project_root: Path) -> None:
        """Setup Python path for module imports."""
        project_root_str = str(project_root)
        
        if project_root_str not in sys.path:
            sys.path.insert(0, project_root_str)
            logger.debug(f"Added to Python path: {project_root_str}")
    
    @staticmethod
    def _import_trainer():
        """Import the CGNETTrainer class with fallback strategies."""
        try:
            from cgnet.trainers import CGNETTrainer
            logger.debug("Successfully imported CGNETTrainer from cgnet.trainers")
            return CGNETTrainer
        except ImportError:
            try:
                # Fallback import strategy
                from trainers import CGNETTrainer
                logger.debug("Successfully imported CGNETTrainer from trainers (fallback)")
                return CGNETTrainer
            except ImportError as e:
                raise ImportError(f"Could not import CGNETTrainer: {str(e)}") from e
    
    def get_job_status(self, job: Any) -> SlurmJobInfo:
        """
        Get detailed status information for a specific job.
        
        Args:
            job: submitit job object
            
        Returns:
            SlurmJobInfo object containing job details
        """
        return SlurmJobMonitor.get_job_info(job)
    
    @staticmethod
    def monitor_jobs(jobs: List[Any]) -> Dict[str, Any]:
        """
        Monitor status of submitted jobs.
        
        Args:
            jobs: List of submitit job objects to monitor
            
        Returns:
            Dictionary containing comprehensive monitoring information
            
        Raises:
            SlurmDependencyError: If submitit is not available
        """
        logger.info(f"Monitoring {len(jobs)} SLURM jobs...")
        result = SlurmJobMonitor.monitor_jobs(jobs)
        logger.info(f"Monitoring completed: {result['completed_jobs']}/{result['total_jobs']} jobs finished")
        return result
    
    @staticmethod
    def cancel_jobs(jobs: List[Any]) -> Dict[str, Any]:
        """
        Cancel submitted jobs.
        
        Args:
            jobs: List of submitit job objects to cancel
            
        Returns:
            Dictionary containing cancellation results
            
        Raises:
            SlurmDependencyError: If submitit is not available
        """
        if not SUBMITIT_AVAILABLE:
            raise SlurmDependencyError("submitit is not installed")
        
        logger.info(f"Cancelling {len(jobs)} SLURM jobs...")
        
        cancelled_jobs = []
        failed_cancellations = []
        
        for job in jobs:
            try:
                job.cancel()
                cancelled_jobs.append(job.job_id)
                logger.info(f"Cancelled job {job.job_id}")
            except Exception as e:
                error_msg = f"Failed to cancel job {job.job_id}: {str(e)}"
                logger.error(error_msg)
                failed_cancellations.append((job.job_id, str(e)))
        
        result = {
            'total_jobs': len(jobs),
            'cancelled_jobs': len(cancelled_jobs),
            'failed_cancellations': len(failed_cancellations),
            'cancelled_job_ids': cancelled_jobs,
            'cancellation_errors': failed_cancellations
        }
        
        logger.info(f"Cancellation completed: {len(cancelled_jobs)}/{len(jobs)} jobs cancelled")
        
        return result
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """
        Get information about the SLURM cluster configuration.
        
        Returns:
            Dictionary containing cluster configuration information
        """
        slurm_config = self.config.get('slurm', {})
        
        info = {
            'slurm_enabled': self._is_slurm_enabled(),
            'submitit_available': SUBMITIT_AVAILABLE,
            'configuration': {
                'partition': slurm_config.get('partition', 'Not specified'),
                'nodes': slurm_config.get('nodes', SlurmConstants.DEFAULT_NODES),
                'cpus_per_task': slurm_config.get('cpus_per_task', SlurmConstants.DEFAULT_CPUS_PER_TASK),
                'mem_gb': slurm_config.get('mem_gb', SlurmConstants.DEFAULT_MEM_GB),
                'gpus_per_node': slurm_config.get('gpus_per_node', 0),
                'time_limit': slurm_config.get('time', 'Not specified'),
                'account': slurm_config.get('account', 'Not specified'),
                'qos': slurm_config.get('qos', 'Not specified')
            }
        }
        
        return info
    
    def reset(self) -> None:
        """Reset the SLURM manager state."""
        logger.info("Resetting SlurmManager state...")
        self._executor = None
        logger.info("SlurmManager state reset successfully") 