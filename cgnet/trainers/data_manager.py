#!/usr/bin/env python
"""Data management for CG-NET training pipeline.

This module provides comprehensive data management capabilities including:
- Raw data loading and preprocessing
- Dataset generation and caching
- Data splitting and cross-validation
- DataLoader preparation for training, validation, testing, and prediction
- Error handling and data validation

The DataManager class serves as the central hub for all data-related operations
in the CG-NET training pipeline.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
from enum import Enum
import logging

import pandas as pd
import torch
from ase.io import read
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from torch_geometric.loader import DataLoader

from ..utils import CGNETFeatureizer, CGNETDataset

# Configure logging
logger = logging.getLogger(__name__)


class SplitType(Enum):
    """Enumeration of data split types."""
    TRAIN_VAL_TEST = "train_val_test"
    TRAIN_TEST = "train_test"
    TRAIN_VAL = "train_val"
    TRAIN_ONLY = "train_only"


class DataConstants:
    """Constants used throughout data management."""
    
    # Default values
    DEFAULT_BATCH_SIZE = 64
    DEFAULT_NUM_WORKERS = 0
    DEFAULT_SAVE_DIR = "./cgnet_dataset"
    DEFAULT_SEED = 42
    DEFAULT_FOLDS = 5
    
    # File names
    DATA_INDEX_FILE = "id_prop_index.csv"
    PROCESSED_DATA_FILE = "data.pt"
    TRAJECTORY_EXTENSION = ".traj"
    
    # CSV columns
    CSV_COLUMNS = ["id", "label", "cidxs"]
    
    # Display limits
    MAX_MISSING_FILES_DISPLAY = 5


class DataValidationError(Exception):
    """Exception raised for data validation errors."""
    pass


class DataLoadingError(Exception):
    """Exception raised for data loading errors."""
    pass


class DataSplitter:
    """Handles data splitting logic for training, validation, and testing."""
    
    @staticmethod
    def determine_split_type(val_ratio: float, test_ratio: float) -> SplitType:
        """
        Determine the type of data split based on ratios.
        
        Args:
            val_ratio: Validation data ratio
            test_ratio: Test data ratio
            
        Returns:
            SplitType enum indicating the split configuration
        """
        has_val = val_ratio > 0
        has_test = test_ratio > 0
        
        if has_val and has_test:
            return SplitType.TRAIN_VAL_TEST
        elif has_test:
            return SplitType.TRAIN_TEST
        elif has_val:
            return SplitType.TRAIN_VAL
        else:
            return SplitType.TRAIN_ONLY
    
    @staticmethod
    def split_dataset_indices(indices: List[int], 
                            val_ratio: float, 
                            test_ratio: float, 
                            seed: int) -> Tuple[List[int], List[int], List[int]]:
        """
        Split dataset indices into train, validation, and test sets.
        
        Args:
            indices: List of dataset indices
            val_ratio: Validation data ratio
            test_ratio: Test data ratio
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_indices, val_indices, test_indices)
        """
        split_type = DataSplitter.determine_split_type(val_ratio, test_ratio)
        
        if split_type == SplitType.TRAIN_ONLY:
            return indices, [], []
        
        elif split_type == SplitType.TRAIN_TEST:
            train_indices, test_indices = train_test_split(
                indices, test_size=test_ratio, random_state=seed
            )
            return train_indices, [], test_indices
        
        elif split_type == SplitType.TRAIN_VAL:
            train_indices, val_indices = train_test_split(
                indices, test_size=val_ratio, random_state=seed
            )
            return train_indices, val_indices, []
        
        elif split_type == SplitType.TRAIN_VAL_TEST:
            train_indices, temp_indices = train_test_split(
                indices, test_size=(val_ratio + test_ratio), random_state=seed
            )
            val_indices, test_indices = train_test_split(
                temp_indices, 
                test_size=test_ratio / (val_ratio + test_ratio), 
                random_state=seed
            )
            return train_indices, val_indices, test_indices
        
        else:
            raise ValueError(f"Unknown split type: {split_type}")


class CrossValidationSplitter:
    """Handles k-fold cross-validation splitting logic."""
    
    @staticmethod
    def create_kfold_splits(dataset: Any,
                          indices: List[int],
                          n_folds: int,
                          stratified: bool,
                          shuffle: bool,
                          seed: int,
                          task_type: str) -> List[Tuple[List[int], List[int]]]:
        """
        Create k-fold cross-validation splits.
        
        Args:
            dataset: Dataset object for accessing labels
            indices: Dataset indices to split
            n_folds: Number of folds
            stratified: Whether to use stratified splitting
            shuffle: Whether to shuffle data
            seed: Random seed for reproducibility
            task_type: Task type ('classification' or 'regression')
            
        Returns:
            List of (train_indices, val_indices) tuples
        """
        if stratified and task_type == 'classification':
            # Extract labels for stratified splitting
            labels = CrossValidationSplitter._extract_labels(dataset, indices)
            kfold = StratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=seed)
            splits = list(kfold.split(indices, labels))
        else:
            kfold = KFold(n_splits=n_folds, shuffle=shuffle, random_state=seed)
            splits = list(kfold.split(indices))
        
        # Convert indices
        fold_splits = []
        for train_idx, val_idx in splits:
            train_indices = [indices[i] for i in train_idx]
            val_indices = [indices[i] for i in val_idx]
            fold_splits.append((train_indices, val_indices))
        
        return fold_splits
    
    @staticmethod
    def _extract_labels(dataset: Any, indices: List[int]) -> List[int]:
        """Extract labels from dataset for stratified splitting."""
        labels = []
        for idx in indices:
            data_point = dataset[idx]
            label = data_point.y.item() if hasattr(data_point.y, 'item') else data_point.y
            labels.append(int(label))
        return labels


class DataManager:
    """Manages data loading, processing, and preparation for CG-NET training.
    
    This class provides a comprehensive interface for data management including:
    - Raw data loading and validation
    - Dataset generation with caching
    - Data splitting for training/validation/testing
    - Cross-validation support
    - DataLoader preparation for all phases
    
    Attributes:
        config (Dict[str, Any]): Configuration dictionary
        dataset (Optional[CGNETDataset]): Cached dataset instance
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DataManager with configuration.
        
        Args:
            config: Configuration dictionary containing data, training, and other settings
            
        Raises:
            ValueError: If required configuration sections are missing
        """
        self.config = config
        self.dataset: Optional[CGNETDataset] = None
        
        # Validate configuration
        self._validate_config()
        
        logger.info("DataManager initialized successfully")
    
    def _validate_config(self) -> None:
        """Validate that required configuration sections exist."""
        required_sections = ['data', 'experiment', 'featurizer']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: '{section}'")
    
    def _is_main_process(self) -> bool:
        """Check if this is the main process in distributed training."""
        return (not hasattr(torch.distributed, 'is_initialized') or 
                not torch.distributed.is_initialized() or 
                torch.distributed.get_rank() == 0)
    
    def load_raw_data(self) -> Tuple[List[str], List[List[int]], List[Any], List[float]]:
        """
        Load raw data from CSV and trajectory files.
        
        Returns:
            Tuple of (ids, cidxs, structures, energies)
            
        Raises:
            DataLoadingError: If data files cannot be loaded
            FileNotFoundError: If required data files are missing
        """
        data_path = self.config['data']['path']
        logger.info(f"Loading raw data from {data_path}...")
        
        try:
            # Load CSV data
            ids, cidxs, energies = self._load_csv_data(data_path)
            
            # Load trajectory structures
            structures, missing_files = self._load_trajectory_files(data_path, ids)
            
            # Handle missing files
            if missing_files:
                ids, cidxs, energies = self._filter_missing_data(
                    ids, cidxs, energies, missing_files
                )
            
            # Log summary
            self._log_data_loading_summary(len(ids), len(missing_files))
            
            return ids, cidxs, structures, energies
            
        except Exception as e:
            raise DataLoadingError(f"Failed to load raw data: {str(e)}") from e
    
    def _load_csv_data(self, data_path: str) -> Tuple[List[str], List[List[int]], List[float]]:
        """Load and parse CSV data file."""
        csv_path = Path(data_path) / DataConstants.DATA_INDEX_FILE
        
        if not csv_path.exists():
            raise FileNotFoundError(f"Data index file not found: {csv_path}")
        
        try:
            df = pd.read_csv(csv_path, names=DataConstants.CSV_COLUMNS, header=0)
        except Exception as e:
            raise DataLoadingError(f"Failed to read CSV file {csv_path}: {str(e)}") from e
        
        ids, cidxs, energies = [], [], []
        
        for _, row in df.iterrows():
            try:
                structure_id = str(row["id"])
                energy = float(row["label"])
                cidx_list = [int(i) for i in str(row["cidxs"]).split(",")]
                
                ids.append(structure_id)
                energies.append(energy)
                cidxs.append(cidx_list)
                
            except (ValueError, TypeError) as e:
                logger.warning(f"Skipping invalid row with ID {row.get('id', 'unknown')}: {e}")
                continue
        
        logger.info(f"Loaded {len(ids)} entries from CSV file")
        return ids, cidxs, energies
    
    def _load_trajectory_files(self, data_path: str, ids: List[str]) -> Tuple[List[Any], List[str]]:
        """Load trajectory files for given IDs."""
        structures = []
        missing_files = []
        
        data_path = Path(data_path)
        
        for structure_id in ids:
            traj_path = data_path / f"{structure_id}{DataConstants.TRAJECTORY_EXTENSION}"
            
            if not traj_path.exists():
                missing_files.append(structure_id)
                continue
            
            try:
                structure = read(str(traj_path))
                structures.append(structure)
            except Exception as e:
                logger.warning(f"Failed to read trajectory file {traj_path}: {e}")
                missing_files.append(structure_id)
        
        return structures, missing_files
    
    def _filter_missing_data(self, ids: List[str], cidxs: List[List[int]], 
                           energies: List[float], missing_files: List[str]) -> Tuple[List[str], List[List[int]], List[float]]:
        """Filter out data entries with missing trajectory files."""
        logger.warning(f"{len(missing_files)} trajectory files not found, removing from dataset")
        
        # Show some missing files for debugging
        files_to_show = missing_files[:DataConstants.MAX_MISSING_FILES_DISPLAY]
        for missing_id in files_to_show:
            logger.warning(f"  - Missing: {missing_id}{DataConstants.TRAJECTORY_EXTENSION}")
        
        if len(missing_files) > DataConstants.MAX_MISSING_FILES_DISPLAY:
            logger.warning(f"  - ... and {len(missing_files) - DataConstants.MAX_MISSING_FILES_DISPLAY} more")
        
        # Filter valid entries
        missing_set = set(missing_files)
        valid_indices = [i for i, structure_id in enumerate(ids) if structure_id not in missing_set]
        
        filtered_ids = [ids[i] for i in valid_indices]
        filtered_cidxs = [cidxs[i] for i in valid_indices]
        filtered_energies = [energies[i] for i in valid_indices]
        
        return filtered_ids, filtered_cidxs, filtered_energies
    
    def _log_data_loading_summary(self, final_count: int, missing_count: int) -> None:
        """Log summary of data loading process."""
        logger.info("Data loading summary:")
        logger.info(f"  - Final dataset size: {final_count}")
        if missing_count > 0:
            logger.info(f"  - Structures filtered (missing files): {missing_count}")
    
    def create_featurizer(self) -> CGNETFeatureizer:
        """
        Create featurizer based on configuration.
        
        Returns:
            Configured CGNETFeatureizer instance
            
        Raises:
            ValueError: If featurizer configuration is invalid
        """
        featurizer_config = self.config['featurizer']
        
        try:
            return CGNETFeatureizer(
                method=featurizer_config['method'],
                neighbor_radius=featurizer_config['neighbor_radius'],
                max_neighbors=featurizer_config['max_neighbors'],
                cluster_radius=featurizer_config.get('cluster_radius'),
                neighbor_depth=featurizer_config.get('neighbor_depth'),
                max_cluster_nodes=featurizer_config.get('max_cluster_nodes'),
                neighbor_mult=featurizer_config['neighbor_mult'],
                tag=featurizer_config['tag'],
                step=featurizer_config['step'],
                with_pseudonodes=featurizer_config['with_pseudonodes'],
                max_distance_factor=featurizer_config.get('max_distance_factor', 2.0),
                small_lattice_threshold=featurizer_config.get('small_lattice_threshold', 5.0),
                enable_smart_images=featurizer_config.get('enable_smart_images', True),
                conservative_small_lattice=featurizer_config.get('conservative_small_lattice', False),
                json_path=featurizer_config.get('json_path'),
                json_file=featurizer_config.get('json_file', 'atom_init.json')
            )
        except Exception as e:
            raise ValueError(f"Failed to create featurizer: {str(e)}") from e
    
    def generate_dataset(self) -> CGNETDataset:
        """
        Generate PyG dataset from raw data with intelligent caching.
        
        Returns:
            Generated or loaded CGNETDataset instance
            
        Raises:
            DataLoadingError: If dataset generation fails
        """
        is_main_process = self._is_main_process()
        
        if is_main_process:
            logger.info("=" * 50)
            logger.info("GENERATING DATASET")
            logger.info("=" * 50)
        
        try:
            data_config = self.config['data']
            save_dir = data_config.get('save_dir', DataConstants.DEFAULT_SAVE_DIR)
            force_reload = data_config.get('force_reload', False)
            
            # Check for existing processed dataset
            processed_path = Path(save_dir) / "processed" / DataConstants.PROCESSED_DATA_FILE
            cache_exists = processed_path.exists()
            
            if cache_exists and not force_reload:
                dataset = self._load_cached_dataset(save_dir, is_main_process)
            else:
                dataset = self._generate_new_dataset(save_dir, force_reload, is_main_process, processed_path)
            
            if is_main_process:
                logger.info(f"Dataset ready with {len(dataset)} samples")
            
            self.dataset = dataset
            return dataset
            
        except Exception as e:
            raise DataLoadingError(f"Failed to generate dataset: {str(e)}") from e
    
    def _load_cached_dataset(self, save_dir: str, is_main_process: bool) -> CGNETDataset:
        """Load dataset from existing cache."""
        if is_main_process:
            logger.info("Found existing processed dataset, loading from cache...")
        
        return CGNETDataset(root=save_dir)
    
    def _generate_new_dataset(self, save_dir: str, force_reload: bool, 
                            is_main_process: bool, processed_path: Path) -> CGNETDataset:
        """Generate new dataset from raw data."""
        if is_main_process:
            if force_reload:
                logger.info("Force reload requested, regenerating dataset from raw data...")
                if processed_path.exists():
                    processed_path.unlink()
            else:
                logger.info("No existing cache found, generating dataset from raw data...")
        
        # Load raw data and create featurizer
        ids, cidxs, structures, energies = self.load_raw_data()
        featurizer = self.create_featurizer()
        
        data_config = self.config['data']
        return CGNETDataset(
            root=save_dir,
            ids=ids,
            cidxs=cidxs,
            structures=structures,
            labels=energies,
            featureizer=featurizer,
            force_reload=force_reload,
            filter_isolated_nodes=data_config.get('filter_isolated_nodes', True)
        )
    
    def prepare_all_dataloaders(self, dataset: Optional[CGNETDataset] = None) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
        """
        Prepare train, validation, and test dataloaders.
        
        Args:
            dataset: Dataset to use. If None, uses cached dataset
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
            
        Raises:
            ValueError: If no dataset is available
        """
        dataset = self._get_dataset(dataset)
        
        logger.info("Preparing dataloaders...")
        
        # Perform data splitting
        train_indices, val_indices, test_indices = self._split_dataset(dataset)
        
        # Create data subsets
        train_dataset, val_dataset, test_dataset = self._create_data_subsets(
            dataset, train_indices, val_indices, test_indices
        )
        
        # Log split summary
        self._log_split_summary(train_dataset, val_dataset, test_dataset)
        
        # Create dataloaders
        train_loader = self._create_dataloader(train_dataset, shuffle=True)
        val_loader = self._create_dataloader(val_dataset, shuffle=False) if val_dataset else None
        test_loader = self._create_dataloader(test_dataset, shuffle=False) if test_dataset else None
        
        return train_loader, val_loader, test_loader
    
    def _get_dataset(self, dataset: Optional[CGNETDataset]) -> CGNETDataset:
        """Get dataset, using provided or cached instance."""
        if dataset is None:
            if self.dataset is None:
                raise ValueError("No dataset available. Generate dataset first.")
            return self.dataset
        return dataset
    
    def _split_dataset(self, dataset: CGNETDataset) -> Tuple[List[int], List[int], List[int]]:
        """Split dataset into train, validation, and test indices."""
        data_config = self.config['data']
        dataset_indices = list(range(len(dataset)))
        
        val_ratio = data_config.get('val_ratio', 0.0)
        test_ratio = data_config.get('test_ratio', 0.0)
        seed = self.config['experiment']['seed']
        
        return DataSplitter.split_dataset_indices(dataset_indices, val_ratio, test_ratio, seed)
    
    def _create_data_subsets(self, dataset: CGNETDataset, 
                           train_indices: List[int], 
                           val_indices: List[int], 
                           test_indices: List[int]) -> Tuple[Any, Optional[Any], Optional[Any]]:
        """Create data subsets from indices."""
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices) if val_indices else None
        test_dataset = torch.utils.data.Subset(dataset, test_indices) if test_indices else None
        
        return train_dataset, val_dataset, test_dataset
    
    def _log_split_summary(self, train_dataset: Any, val_dataset: Optional[Any], test_dataset: Optional[Any]) -> None:
        """Log dataset split summary."""
        val_size = len(val_dataset) if val_dataset else 0
        test_size = len(test_dataset) if test_dataset else 0
        logger.info(f"Dataset split: Train={len(train_dataset)}, Val={val_size}, Test={test_size}")
    
    def prepare_kfold_splits(self, dataset: Optional[CGNETDataset] = None, 
                           reserve_test: bool = True) -> Tuple[List[Tuple[DataLoader, DataLoader]], Optional[DataLoader]]:
        """
        Prepare k-fold cross-validation splits.
        
        Args:
            dataset: Dataset to use. If None, uses cached dataset
            reserve_test: Whether to reserve a portion of data for final testing
            
        Returns:
            Tuple of (fold_loaders, test_loader) where fold_loaders is a list of 
            (train_loader, val_loader) pairs
            
        Raises:
            ValueError: If no dataset is available or CV configuration is invalid
        """
        dataset = self._get_dataset(dataset)
        
        logger.info("Preparing k-fold cross-validation splits...")
        
        cv_config = self.config.get('cross_validation', {})
        self._validate_cv_config(cv_config)
        
        # Prepare indices for cross-validation
        train_val_indices, test_indices = self._prepare_cv_indices(dataset, reserve_test)
        
        # Create k-fold splits
        fold_splits = self._create_cv_splits(dataset, train_val_indices, cv_config)
        
        # Create test loader if needed
        test_loader = self._create_cv_test_loader(dataset, test_indices)
        
        # Create fold loaders
        fold_loaders = self._create_cv_fold_loaders(dataset, fold_splits)
        
        return fold_loaders, test_loader
    
    def _validate_cv_config(self, cv_config: Dict[str, Any]) -> None:
        """Validate cross-validation configuration."""
        n_folds = cv_config.get('n_folds', DataConstants.DEFAULT_FOLDS)
        if n_folds < 2:
            raise ValueError(f"Number of folds must be at least 2, got {n_folds}")
    
    def _prepare_cv_indices(self, dataset: CGNETDataset, reserve_test: bool) -> Tuple[List[int], Optional[List[int]]]:
        """Prepare indices for cross-validation, optionally reserving test set."""
        dataset_indices = list(range(len(dataset)))
        test_indices = None
        
        if reserve_test:
            data_config = self.config['data']
            test_ratio = data_config.get('test_ratio', 0.0)
            
            if test_ratio > 0:
                seed = self.config['experiment']['seed']
                cv_config = self.config.get('cross_validation', {})
                shuffle = cv_config.get('shuffle', True)
                
                train_val_indices, test_indices = train_test_split(
                    dataset_indices,
                    test_size=test_ratio,
                    random_state=seed,
                    shuffle=shuffle
                )
            else:
                train_val_indices = dataset_indices
        else:
            train_val_indices = dataset_indices
        
        return train_val_indices, test_indices
    
    def _create_cv_splits(self, dataset: CGNETDataset, train_val_indices: List[int], 
                         cv_config: Dict[str, Any]) -> List[Tuple[List[int], List[int]]]:
        """Create cross-validation splits."""
        n_folds = cv_config.get('n_folds', DataConstants.DEFAULT_FOLDS)
        stratified = cv_config.get('stratified', False)
        shuffle = cv_config.get('shuffle', True)
        seed = self.config['experiment']['seed']
        task_type = self.config['model']['task']
        
        return CrossValidationSplitter.create_kfold_splits(
            dataset, train_val_indices, n_folds, stratified, shuffle, seed, task_type
        )
    
    def _create_cv_test_loader(self, dataset: CGNETDataset, test_indices: Optional[List[int]]) -> Optional[DataLoader]:
        """Create test loader for cross-validation."""
        if test_indices is not None and len(test_indices) > 0:
            test_dataset = torch.utils.data.Subset(dataset, test_indices)
            test_loader = self._create_dataloader(test_dataset, shuffle=False)
            logger.info(f"Reserved test set: {len(test_dataset)} samples")
            return test_loader
        return None
    
    def _create_cv_fold_loaders(self, dataset: CGNETDataset, 
                               fold_splits: List[Tuple[List[int], List[int]]]) -> List[Tuple[DataLoader, DataLoader]]:
        """Create dataloaders for each cross-validation fold."""
        fold_loaders = []
        
        for fold_idx, (train_indices, val_indices) in enumerate(fold_splits):
            train_dataset = torch.utils.data.Subset(dataset, train_indices)
            val_dataset = torch.utils.data.Subset(dataset, val_indices)
            
            train_loader = self._create_dataloader(train_dataset, shuffle=True)
            val_loader = self._create_dataloader(val_dataset, shuffle=False)
            
            fold_loaders.append((train_loader, val_loader))
            logger.info(f"Fold {fold_idx + 1}: Train={len(train_dataset)}, Val={len(val_dataset)}")
        
        return fold_loaders
    
    def create_dataloader(self, dataset: CGNETDataset, shuffle: bool = False) -> DataLoader:
        """
        Create a single DataLoader for the given dataset.
        
        Args:
            dataset: Dataset to create loader for
            shuffle: Whether to shuffle the data
            
        Returns:
            Configured DataLoader instance
        """
        return self._create_dataloader(dataset, shuffle)
    
    def _create_dataloader(self, dataset: Any, shuffle: bool) -> DataLoader:
        """Internal method to create dataloaders with consistent configuration."""
        if dataset is None:
            return None
        
        training_config = self.config.get('training', {})
        batch_size = training_config.get('batch_size', DataConstants.DEFAULT_BATCH_SIZE)
        num_workers = training_config.get('num_workers', DataConstants.DEFAULT_NUM_WORKERS)
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
        )
    
    def prepare_prediction_dataloader(self, dataset: Optional[CGNETDataset] = None, 
                                    batch_size: Optional[int] = None) -> DataLoader:
        """
        Prepare a single dataloader containing all data for prediction.
        
        Args:
            dataset: Dataset to use. If None, uses cached dataset
            batch_size: Batch size for prediction. If None, uses config values
            
        Returns:
            DataLoader containing all data for prediction
            
        Raises:
            ValueError: If no dataset is available
        """
        dataset = self._get_dataset(dataset)
        
        # Determine batch size with priority order
        if batch_size is None:
            batch_size = self._get_prediction_batch_size()
        
        # Determine number of workers
        num_workers = self._get_prediction_num_workers()
        
        logger.info(f"Preparing prediction dataloader for {len(dataset)} samples...")
        
        prediction_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,  # No need to shuffle for prediction
        )
        
        logger.info(f"Prediction dataloader prepared (batch_size={batch_size}, num_workers={num_workers})")
        
        return prediction_loader
    
    def _get_prediction_batch_size(self) -> int:
        """Get batch size for prediction with fallback logic."""
        prediction_config = self.config.get('prediction', {})
        training_config = self.config.get('training', {})
        
        return (prediction_config.get('batch_size') or 
                training_config.get('batch_size', DataConstants.DEFAULT_BATCH_SIZE))
    
    def _get_prediction_num_workers(self) -> int:
        """Get number of workers for prediction with fallback logic."""
        prediction_config = self.config.get('prediction', {})
        training_config = self.config.get('training', {})
        
        return (prediction_config.get('num_workers') or 
                training_config.get('num_workers', DataConstants.DEFAULT_NUM_WORKERS))
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the current dataset.
        
        Returns:
            Dictionary containing dataset information
            
        Raises:
            ValueError: If no dataset is available
        """
        if self.dataset is None:
            raise ValueError("No dataset available. Generate dataset first.")
        
        return {
            'size': len(self.dataset),
            'data_path': self.config['data']['path'],
            'save_dir': self.config['data'].get('save_dir', DataConstants.DEFAULT_SAVE_DIR),
            'featurizer_method': self.config['featurizer']['method'],
            'task_type': self.config['model']['task']
        }
    
    def reset(self) -> None:
        """Reset the data manager state."""
        logger.info("Resetting DataManager state...")
        self.dataset = None
        logger.info("DataManager state reset successfully") 