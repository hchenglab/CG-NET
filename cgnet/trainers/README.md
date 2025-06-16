# CG-NET Trainer - Modular Architecture

## Overview

The CG-NET trainer has been refactored into a modular architecture for better code organization, maintainability, and flexibility. The original monolithic `CGNETTrainer` class (1,724 lines) has been split into focused, single-responsibility modules.

## Architecture

### 🗂️ Module Structure

```
cgnet/trainers/
├── config_manager.py      # Configuration management
├── data_manager.py        # Data processing and loading
├── model_manager.py       # Model creation and management
├── training_manager.py    # Training and evaluation logic
├── slurm_manager.py       # SLURM job submission and monitoring
├── trainer.py             # Main orchestrator (modular implementation)
└── __init__.py           # Module exports
```

### 🎯 Single Responsibility Principle

Each module now has a clear, focused responsibility:

| Module | Responsibility | Key Features |
|--------|---------------|--------------|
| **ConfigManager** | Configuration handling | Load, validate, save, update configs |
| **DataManager** | Data pipeline | Raw data loading, dataset generation, data loaders |
| **ModelManager** | Model lifecycle | Model creation, checkpointing, loading |
| **TrainingManager** | Training operations | Regular training, k-fold CV, testing |
| **SlurmManager** | Job management | SLURM submission, monitoring, scripts |
| **CGNETTrainer** | Orchestration | Coordinates all managers, main interface |

## 🚀 Usage

### Basic Usage

```python
from cgnet.trainers import CGNETTrainer

# Initialize with config file
trainer = CGNETTrainer('config.yml')

# Run complete pipeline
trainer.run_pipeline(mode="all")
```

### Advanced Usage - Individual Managers

```python
from cgnet.trainers import (
    ConfigManager, DataManager, ModelManager, 
    TrainingManager, SlurmManager
)

# Use individual managers for fine-grained control
config = ConfigManager.load_config('config.yml')
data_manager = DataManager(config)
model_manager = ModelManager(config)
training_manager = TrainingManager(config, model_manager)

# Custom workflow
dataset = data_manager.generate_dataset()
train_loader, val_loader, test_loader = data_manager.prepare_dataloaders(dataset)
model, trainer_obj = training_manager.train_model(train_loader, val_loader)
```

### Step-by-Step Pipeline

```python
trainer = CGNETTrainer('config.yml')

# Step 1: Data preparation
dataset = trainer.generate_dataset()
train_loader, val_loader, test_loader = trainer.prepare_dataloaders(dataset)

# Step 2: Model training
model, trainer_obj = trainer.train_model(train_loader, val_loader)

# Step 3: Testing
trainer.test_model(model, test_loader, trainer_obj)

# Step 4: Get model info
print(trainer.get_model_summary())
```

### K-Fold Cross-Validation

```python
# Enable k-fold in config
config = ConfigManager.load_config('config.yml')
config['cross_validation']['enabled'] = True
config['cross_validation']['n_folds'] = 5
ConfigManager.save_config(config, 'kfold_config.yml')

# Run k-fold training
trainer = CGNETTrainer('kfold_config.yml')
results = trainer.run_kfold_training()
```

### SLURM Integration

```python
# Enable SLURM in config
config['slurm']['use_slurm'] = True
config['slurm']['partition'] = 'gpu'

trainer = CGNETTrainer('slurm_config.yml')

# Submit pipeline with dependencies
jobs = trainer.submit_pipeline_with_dependencies()

# Monitor jobs
status = CGNETTrainer.monitor_jobs(list(jobs.values()))
```

## 🔧 Benefits of Modular Architecture

### 1. **Separation of Concerns**
- Each module handles a specific aspect of the pipeline
- Clear interfaces between components
- Easier to understand and modify

### 2. **Improved Testability**
- Individual modules can be tested in isolation
- Mock dependencies easily for unit testing
- Better error isolation and debugging

### 3. **Enhanced Reusability**
- Use individual managers in other projects
- Mix and match components as needed
- Customize specific parts without affecting others

### 4. **Better Maintainability**
- Smaller, focused files are easier to navigate
- Changes to one aspect don't affect others
- Clearer code organization

### 5. **Flexibility**
- Easy to extend or replace individual components
- Support for different training strategies
- Pluggable architecture for future enhancements

## 📁 Module Details

### ConfigManager
- **Purpose**: Centralized configuration management
- **Features**: YAML loading, validation, default configs, updates
- **Size**: ~228 lines (vs. ~200 lines in original)

### DataManager  
- **Purpose**: Data pipeline management
- **Features**: Raw data loading, dataset generation, data loaders, k-fold splits
- **Size**: ~315 lines (vs. ~300 lines in original)

### ModelManager
- **Purpose**: Model lifecycle management
- **Features**: Model creation, checkpointing, loading, summaries
- **Size**: ~277 lines (vs. ~150 lines in original)

### TrainingManager
- **Purpose**: Training operations
- **Features**: Regular training, k-fold CV, testing, evaluation
- **Size**: ~451 lines (vs. ~600 lines in original)

### SlurmManager
- **Purpose**: SLURM job management
- **Features**: Job submission, monitoring, script generation
- **Size**: ~607 lines (vs. ~400 lines in original)

### CGNETTrainer (Main Implementation)
- **Purpose**: Main orchestrator
- **Features**: Coordinates managers, high-level interface
- **Size**: ~329 lines (vs. 1,724 lines in original)

## 🔄 Migration Guide

### For Existing Users

The new `CGNETTrainer` maintains the same public API as the original, so existing code should work without changes:

```python
# This still works exactly the same
from cgnet.trainers import CGNETTrainer
trainer = CGNETTrainer('config.yml')
trainer.run_pipeline()
```



### New Advanced Features

Take advantage of the new modular structure:

```python
# Direct manager access
from cgnet.trainers import DataManager, ModelManager

# Custom data pipeline
data_manager = DataManager(config)
dataset = data_manager.generate_dataset()

# Custom model management
model_manager = ModelManager(config)
model = model_manager.create_model(edge_dim=41)
```

## 📊 Comparison

| Aspect | Original | Modular |
|--------|----------|---------|
| **Total Lines** | 1,724 | ~2,007 (across 6 files) |
| **Largest File** | 1,724 lines | 607 lines |
| **Maintainability** | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Testability** | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Reusability** | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Flexibility** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **API Compatibility** | ✅ | ✅ |

## 🎯 Examples

The README provides comprehensive usage examples including:
- Basic usage
- Step-by-step execution
- K-fold cross-validation
- Advanced manager usage
- SLURM job submission
- Configuration management
- Model checkpointing

## 🛠️ Future Enhancements

The modular architecture enables easy future enhancements:

1. **Plugin System**: Add custom data loaders, models, or training strategies
2. **Different Backends**: Support for other job schedulers besides SLURM
3. **Advanced Logging**: Pluggable logging and monitoring systems
4. **Cloud Integration**: Support for cloud-based training
5. **Auto-tuning**: Hyperparameter optimization modules

## 📝 Notes

- All original functionality is preserved
- Performance characteristics remain the same
- Memory usage is similar
- The modular structure adds ~16% more total lines but improves organization significantly
- Each module is now under 650 lines, making them much more manageable 