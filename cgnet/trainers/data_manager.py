#!/usr/bin/env python
"""Data management for CG-NET training pipeline."""

import os
import pandas as pd
import torch
from typing import Dict, Any, List, Tuple, Optional
from ase.io import read
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from torch_geometric.loader import DataLoader

from ..utils import CGNETFeatureizer, CGNETDataset


class DataManager:
    """Manages data loading, processing, and preparation for CG-NET training."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DataManager with configuration.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary
        """
        self.config = config
        self.dataset = None
    
    def load_raw_data(self) -> Tuple[list, list, list, list]:
        """Load raw data from CSV and trajectory files."""
        data_path = self.config['data']['path']
        print(f"Loading raw data from {data_path}...")
        
        ids, cidxs, energies = [], [], []
        csv_path = os.path.join(data_path, "id_prop_index.csv")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Data index file not found: {csv_path}")
        
        df = pd.read_csv(csv_path, names=["id", "label", "cidxs"], header=0)
        
        for _, row in df.iterrows():
            structure_id = str(row["id"])
            ids.append(structure_id)
            energies.append(float(row["label"]))
            cidxs.append([int(i) for i in str(row["cidxs"]).split(",")])

        # Load trajectory files
        structures = []
        missing_files = []
        for id in ids:
            traj_path = os.path.join(data_path, f"{id}.traj")
            if not os.path.exists(traj_path):
                missing_files.append(id)
                continue
            structures.append(read(traj_path))
        
        # Handle missing files
        if missing_files:
            print(f"Warning: {len(missing_files)} trajectory files not found, removing from dataset:")
            for missing_id in missing_files[:5]:  # Show first 5 missing files
                print(f"  - {missing_id}.traj")
            if len(missing_files) > 5:
                print(f"  - ... and {len(missing_files) - 5} more")
            
            # Remove missing entries
            valid_indices = []
            for i, id in enumerate(ids):
                if id not in missing_files:
                    valid_indices.append(i)
            
            ids = [ids[i] for i in valid_indices]
            cidxs = [cidxs[i] for i in valid_indices]
            energies = [energies[i] for i in valid_indices]
        
        print(f"Data loading summary:")
        print(f"  - Total structures in CSV: {len(df)}")
        if missing_files:
            print(f"  - Structures filtered (missing files): {len(missing_files)}")
        print(f"  - Final dataset size: {len(ids)}")
        
        return ids, cidxs, structures, energies
    
    def create_featurizer(self) -> CGNETFeatureizer:
        """Create featurizer based on configuration."""
        featurizer_config = self.config['featurizer']
        
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
    
    def generate_dataset(self) -> CGNETDataset:
        """Generate PyG dataset from raw data."""
        is_main_process = not hasattr(torch.distributed, 'is_initialized') or not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        
        if is_main_process:
            print("=" * 50)
            print("GENERATING DATASET")
            print("=" * 50)
        
        data_config = self.config['data']
        save_dir = data_config['save_dir'] if data_config['save_dir'] else "./cgnet_dataset"
        processed_path = os.path.join(save_dir, "processed", "data.pt")
        cache_exists = os.path.exists(processed_path)
        
        if cache_exists and not data_config['clean_cache']:
            if is_main_process:
                print(f"Found existing processed dataset: {processed_path}")
                print("Loading dataset from cache...")
            dataset = CGNETDataset.from_cache(
                root=save_dir,
                use_parallel=data_config['use_parallel']
            )
        else:
            if is_main_process:
                if data_config['clean_cache']:
                    print("Clean cache requested, will regenerate dataset from raw data...")
                    if cache_exists:
                        os.remove(processed_path)
                else:
                    print("No existing cache found, generating dataset from raw data...")
            ids, cidxs, structures, energies = self.load_raw_data()
            featurizer = self.create_featurizer()
            dataset = CGNETDataset.from_data_lists(
                root=save_dir,
                ids=ids,
                cidxs=cidxs,
                structures=structures,
                labels=energies,
                featureizer=featurizer,
                use_parallel=data_config['use_parallel'],
                force_reload=data_config['clean_cache'],
                filter_isolated_nodes=data_config['filter_isolated_nodes']
            )
        
        if is_main_process:
            print(f"Dataset generated with {len(dataset)} samples")
        
        self.dataset = dataset
        return dataset
    
    def prepare_all_dataloaders(self, dataset: Optional[CGNETDataset] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare train, validation, and test dataloaders."""
        if dataset is None:
            if self.dataset is None:
                raise ValueError("No dataset available. Generate dataset first.")
            dataset = self.dataset
        
        print("Preparing dataloaders...")
        data_config = self.config['data']
        dataset_indices = list(range(len(dataset)))
        
        train_indices, temp_indices = train_test_split(
            dataset_indices,
            test_size=(data_config['val_ratio'] + data_config['test_ratio']),
            random_state=self.config['experiment']['seed']
        )
        val_indices, test_indices = train_test_split(
            temp_indices,
            test_size=data_config['test_ratio'] / (data_config['val_ratio'] + data_config['test_ratio']),
            random_state=self.config['experiment']['seed']
        )
        
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)
        
        print(f"Dataset split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
        
        training_config = self.config['training']
        train_loader = DataLoader(
            train_dataset,
            batch_size=training_config['batch_size'],
            num_workers=training_config['num_workers'],
            shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=training_config['batch_size'],
            num_workers=training_config['num_workers'],
            shuffle=False,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=training_config['batch_size'],
            num_workers=training_config['num_workers'],
            shuffle=False,
        )
        
        return train_loader, val_loader, test_loader
    
    def prepare_kfold_splits(self, dataset: Optional[CGNETDataset] = None, 
                            reserve_test: bool = True) -> Tuple[List[Tuple[DataLoader, DataLoader]], Optional[DataLoader]]:
        """Prepare k-fold cross-validation splits."""
        if dataset is None:
            if self.dataset is None:
                raise ValueError("No dataset available. Generate dataset first.")
            dataset = self.dataset
        
        print("Preparing k-fold cross-validation splits...")
        cv_config = self.config.get('cross_validation', {})
        data_config = self.config['data']
        training_config = self.config['training']
        
        n_folds = cv_config.get('n_folds', 5)
        stratified = cv_config.get('stratified', False)
        shuffle = cv_config.get('shuffle', True)
        dataset_indices = list(range(len(dataset)))
        test_indices = None
        
        if reserve_test:
            test_ratio = data_config['test_ratio']
            if test_ratio > 0:
                train_val_indices, test_indices = train_test_split(
                    dataset_indices,
                    test_size=test_ratio,
                    random_state=self.config['experiment']['seed'],
                    shuffle=shuffle
                )
            else:
                train_val_indices = dataset_indices
        else:
            train_val_indices = dataset_indices
        
        if stratified and self.config['model']['task'] == 'classification':
            labels = []
            for idx in train_val_indices:
                data_point = dataset[idx]
                label = data_point.y.item() if hasattr(data_point.y, 'item') else data_point.y
                labels.append(int(label))
            
            kfold = StratifiedKFold(
                n_splits=n_folds,
                shuffle=shuffle,
                random_state=self.config['experiment']['seed']
            )
            splits = list(kfold.split(train_val_indices, labels))
        else:
            kfold = KFold(
                n_splits=n_folds,
                shuffle=shuffle,
                random_state=self.config['experiment']['seed']
            )
            splits = list(kfold.split(train_val_indices))
        
        fold_splits = []
        for train_idx, val_idx in splits:
            train_indices = [train_val_indices[i] for i in train_idx]
            val_indices = [train_val_indices[i] for i in val_idx]
            fold_splits.append({
                'train_indices': train_indices,
                'val_indices': val_indices
            })
        
        test_loader = None
        if test_indices is not None and len(test_indices) > 0:
            test_dataset = torch.utils.data.Subset(dataset, test_indices)
            test_loader = DataLoader(
                test_dataset,
                batch_size=training_config['batch_size'],
                num_workers=training_config['num_workers'],
                shuffle=False,
            )
            print(f"Reserved test set: {len(test_dataset)} samples")
        
        fold_loaders = []
        for fold_idx, fold_split in enumerate(fold_splits):
            train_indices = fold_split['train_indices']
            val_indices = fold_split['val_indices']
            train_dataset = torch.utils.data.Subset(dataset, train_indices)
            val_dataset = torch.utils.data.Subset(dataset, val_indices)
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=training_config['batch_size'],
                num_workers=training_config['num_workers'],
                shuffle=True,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=training_config['batch_size'],
                num_workers=training_config['num_workers'],
                shuffle=False,
            )
            
            fold_loaders.append((train_loader, val_loader))
            print(f"Fold {fold_idx + 1}: Train={len(train_dataset)}, Val={len(val_dataset)}")
        
        return fold_loaders, test_loader
    
    def create_dataloader(self, dataset: CGNETDataset, shuffle: bool = False) -> DataLoader:
        """Create a single DataLoader for the given dataset."""
        training_config = self.config['training']
        return DataLoader(
            dataset,
            batch_size=training_config['batch_size'],
            num_workers=training_config['num_workers'],
            shuffle=shuffle,
        )
    
    def prepare_prediction_dataloader(self, dataset: Optional[CGNETDataset] = None, batch_size: Optional[int] = None) -> DataLoader:
        """
        Prepare a single dataloader containing all data for prediction.
        
        Parameters
        ----------
        dataset : CGNETDataset, optional
            Dataset to use. If None, uses self.dataset or raises error.
        batch_size : int, optional
            Batch size for prediction. If None, uses config batch_size.
            
        Returns
        -------
        DataLoader
            A single dataloader containing all data for prediction.
        """
        if dataset is None:
            if self.dataset is None:
                raise ValueError("No dataset available. Generate dataset first.")
            dataset = self.dataset
        
        if batch_size is None:
            batch_size = self.config.get('training', {}).get('batch_size', 64)
        
        num_workers = self.config.get('training', {}).get('num_workers', 0)
        
        print(f"Preparing prediction dataloader for {len(dataset)} samples...")
        prediction_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,  # No need to shuffle for prediction
        )
        print(f"Prediction dataloader prepared (batch_size={batch_size})")
        
        return prediction_loader 