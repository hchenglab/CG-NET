# CG-NET Configuration Files

This directory contains configuration files for different CG-NET training and inference scenarios. Each configuration file is optimized for specific use cases and contains detailed comments explaining all parameters.

## Available Configuration Files

### 1. **classification.yml**
**Purpose**: Binary and multi-class classification tasks
**Use Case**: When predicting discrete categories (e.g., structure types, stability classes)

**Key Features**:
- Monitors validation accuracy (`val_acc`) for model selection
- Includes early stopping with patience tuned for classification (20 epochs)
- Stratified k-fold cross-validation by default
- Resource-efficient SLURM settings

**Example Usage**:
```python
from cgnet.trainers import CGNETTrainer

trainer = CGNETTrainer("cgnet/configs/classification.yml")
results = trainer.run_pipeline(mode="all")
```

### 2. **regression.yml**
**Purpose**: Regression tasks (default configuration)
**Use Case**: When predicting continuous values (e.g., formation energy, adsorption energy, mechanical properties)

**Key Features**:
- Monitors validation MAE (`val_mae`) for model selection
- Standard k-fold cross-validation (non-stratified)
- Optimized for continuous target prediction
- Resource-efficient SLURM settings

**Example Usage**:
```python
from cgnet.trainers import CGNETTrainer

trainer = CGNETTrainer("cgnet/configs/regression.yml")
results = trainer.run_pipeline(mode="all")
```

### 3. **prediction.yml**
**Purpose**: Inference on new data using pre-trained models
**Use Case**: Making predictions on new structures with an already trained model

**Key Features**:
- Minimal training configuration (not used for training)
- Optimized batch size (128) for efficient prediction
- Special data splitting (100% test, 0% train/val)
- Conservative parallelization settings (single worker)
- Additional prediction-specific settings (ensemble, uncertainty estimation)
- Reduced resource requirements for SLURM jobs (16GB memory, 4 CPUs)

**Example Usage**:
```python
from cgnet.trainers import CGNETTrainer

trainer = CGNETTrainer("cgnet/configs/prediction.yml")
# Provide path to trained model
results = trainer.run_pipeline(mode="predict", model_path="/path/to/trained/model.ckpt")
```

## Configuration Structure

All configuration files follow the same structure with the following main sections:

### Core Sections

1. **`experiment`**: Basic experiment settings (name, seed)
2. **`data`**: Data loading and preprocessing settings
3. **`model`**: Neural network architecture parameters
4. **`training`**: Training hyperparameters and optimization settings
5. **`cross_validation`**: K-fold cross-validation configuration
6. **`featurizer`**: Graph construction and featurization parameters
7. **`logging`**: Logging and model checkpointing settings
8. **`device`**: Device and precision configuration
9. **`slurm`**: Cluster job submission settings

### Task-Specific Sections

- **`prediction`**: Additional settings for prediction tasks (only in `prediction.yml`)

## Usage Patterns

### 1. Standard Training Workflow
```python
# For regression tasks
trainer = CGNETTrainer("cgnet/configs/regression.yml")
trainer.run_pipeline(mode="all")

# For classification tasks  
trainer = CGNETTrainer("cgnet/configs/classification.yml")
trainer.run_pipeline(mode="all")
```

### 2. Cross-Validation Evaluation
For k-fold cross-validation, enable it in any configuration:
```python
from cgnet.trainers import ConfigManager

# Load and modify config for cross-validation
config = ConfigManager.load_config("cgnet/configs/regression.yml")
config['cross_validation']['enabled'] = True
config['cross_validation']['n_folds'] = 5

# Save and use modified config
ConfigManager.save_config(config, "regression_cv.yml")
trainer = CGNETTrainer("regression_cv.yml")
results = trainer.run_pipeline(mode="all")
```

### 3. Prediction on New Data
```python
trainer = CGNETTrainer("cgnet/configs/prediction.yml")
predictions = trainer.run_pipeline(
    mode="predict", 
    model_path="logs/regression/best_model.ckpt"
)
```

### 4. Custom Configuration
```python
from cgnet.trainers import ConfigManager

# Load and modify existing config
config = ConfigManager.load_config("cgnet/configs/regression.yml")
config['training']['lr'] = 0.01  # Modify learning rate
config['model']['hidden_node_dim'] = 128  # Change model size

# Save custom config
ConfigManager.save_config(config, "my_custom_config.yml")

# Use custom config
trainer = CGNETTrainer("my_custom_config.yml")
```

## SLURM Integration

All configuration files include SLURM settings for cluster computing:

- **Default**: `use_slurm: false` (safe default)
- **Resource allocation**: Optimized for each task type
- **Job arrays**: Support for parallel processing
- **Email notifications**: Configurable completion/failure alerts

To enable SLURM:
```yaml
slurm:
  use_slurm: true
  partition: "your_gpu_partition"
  account: "your_account"  # if required
  mail_user: "your_email@domain.com"  # for notifications
```

## Best Practices

1. **Start with defaults**: Use `regression.yml` or `classification.yml` as starting points
2. **Cross-validation setup**: Enable CV in existing configs rather than separate files
3. **Prediction optimization**: Use `prediction.yml` for inference on large datasets
4. **Resource planning**: Use conservative settings (4 CPUs, single worker) for stability
5. **Reproducibility**: Set deterministic=true and use fixed seeds for reproducible results
6. **Monitoring**: Choose appropriate metrics (`val_mae` for regression, `val_acc` for classification)


### Validation
The configuration manager automatically validates all parameters. Run:
```python
from cgnet.trainers import ConfigManager
ConfigManager.validate_config(ConfigManager.load_config("your_config.yml"))
```

## Version Compatibility

These configuration files are compatible with:
- CG-NET version: Latest
- PyTorch Lightning: ≥1.6.0
- Python: ≥3.8

For older versions, some parameters may need adjustment. 
