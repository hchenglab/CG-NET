from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple, Union
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score

import pytorch_lightning as pl

try:
    from torch_geometric.nn import MessagePassing, global_add_pool
    from torch_geometric.data import Data, Batch
except ImportError:
    raise ImportError("This class requires PyTorch Geometric to be installed.")


# Constants
DEFAULT_IN_NODE_DIM = 92
DEFAULT_HIDDEN_NODE_DIM = 64
DEFAULT_IN_EDGE_DIM = 41
DEFAULT_PREDICTOR_HIDDEN_DIM = 128
DEFAULT_N_CONV = 3
DEFAULT_N_H = 2
DEFAULT_N_TASKS = 1
DEFAULT_N_CLASSES = 2
DEFAULT_LR = 1e-3
DEFAULT_TMAX = 300

SUPPORTED_TASKS = {"classification", "regression"}


class CGNETLayer(MessagePassing):
    """
    CG-NET convolutional layer implementing message passing with gated updates.
    
    This layer performs graph convolution using message passing with gated
    mechanisms to control information flow between nodes.
    """
    
    def __init__(
        self, 
        hidden_node_dim: int, 
        edge_dim: int, 
        batch_norm: bool = True
    ) -> None:
        """
        Initialize the CG-NET layer.
        
        Parameters
        ----------
        hidden_node_dim : int
            The dimension of hidden node feature vectors.
        edge_dim : int
            The dimension of edge feature vectors.
        batch_norm : bool, default=True
            Whether to apply batch normalization.
            
        Raises
        ------
        ValueError
            If dimensions are not positive integers.
        """
        if hidden_node_dim <= 0 or edge_dim <= 0:
            raise ValueError("hidden_node_dim and edge_dim must be positive integers")
            
        super(CGNETLayer, self).__init__(aggr='add')
        
        # Calculate dimensions for linear transformation
        z_dim = 2 * hidden_node_dim + edge_dim
        linear_out_dim = 2 * hidden_node_dim
        
        # Define layers
        self.linear = nn.Linear(z_dim, linear_out_dim)
        self.batch_norm = nn.BatchNorm1d(linear_out_dim) if batch_norm else None

    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """
        Update node representations using message passing.

        Parameters
        ----------
        x : torch.Tensor
            Node features with shape (N, hidden_node_dim).
        edge_index : torch.Tensor
            Edge indices with shape (2, num_edges).
        edge_attr : torch.Tensor
            Edge features with shape (num_edges, edge_dim).

        Returns
        -------
        torch.Tensor
            Updated node features with shape (N, hidden_node_dim).
        """
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(
        self, 
        x_i: torch.Tensor, 
        x_j: torch.Tensor, 
        edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute messages between nodes.
        
        Parameters
        ----------
        x_i : torch.Tensor
            Source node features.
        x_j : torch.Tensor
            Target node features.
        edge_attr : torch.Tensor
            Edge features.
            
        Returns
        -------
        torch.Tensor
            Computed messages.
        """
        # Concatenate node and edge features
        z = torch.cat([x_i, x_j, edge_attr], dim=1)
        z = self.linear(z)
        
        # Apply batch normalization if enabled
        if self.batch_norm is not None:
            z = self.batch_norm(z)
        
        # Split into gating and message components
        gated_z, message_z = z.chunk(2, dim=1)
        gated_z = torch.sigmoid(gated_z)
        message_z = F.softplus(message_z)
        
        return gated_z * message_z

    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Update node features with aggregated messages.
        
        Parameters
        ----------
        aggr_out : torch.Tensor
            Aggregated messages.
        x : torch.Tensor
            Original node features.
            
        Returns
        -------
        torch.Tensor
            Updated node features.
        """
        return x + aggr_out


class CGNETModel(nn.Module):
    """
    Pure PyTorch implementation of CG-NET model without training logic.
    
    This is the core neural network model that can be used independently
    of any training framework (PyTorch Lightning, pure PyTorch, etc.).
    """
    
    def __init__(
        self,
        in_node_dim: int = DEFAULT_IN_NODE_DIM,
        hidden_node_dim: int = DEFAULT_HIDDEN_NODE_DIM,
        in_edge_dim: int = DEFAULT_IN_EDGE_DIM,
        predictor_hidden_dim: int = DEFAULT_PREDICTOR_HIDDEN_DIM,
        n_conv: int = DEFAULT_N_CONV,
        n_h: int = DEFAULT_N_H,
        n_tasks: int = DEFAULT_N_TASKS,
        task: str = "regression",
        n_classes: int = DEFAULT_N_CLASSES,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the CG-NET model.
        
        Parameters
        ----------
        in_node_dim : int, default=92
            Initial node feature vector dimension (based on atom_init.json).
        hidden_node_dim : int, default=64
            Hidden node feature vector dimension.
        in_edge_dim : int, default=41
            Initial edge feature vector dimension (based on CGNETFeatureizer default).
        n_conv : int, default=3
            Number of convolutional layers.
        n_h : int, default=2
            Number of hidden layers in the MLP predictor.
        predictor_hidden_dim : int, default=128
            Hidden layer size in the output MLP predictor.
        n_tasks : int, default=1
            Number of output tasks.
        task : str, default='regression'
            Task type: 'classification' or 'regression'.
        n_classes : int, default=2
            Number of classes (only used in classification mode).
        **kwargs
            Additional arguments (ignored for compatibility).
            
        Raises
        ------
        ValueError
            If task is not supported or parameters are invalid.
        """
        super(CGNETModel, self).__init__()
        
        # Validate inputs
        self._validate_init_params(
            task, n_conv, n_h, n_tasks, n_classes,
            in_node_dim, hidden_node_dim, in_edge_dim, predictor_hidden_dim
        )

        # Store model configuration
        self.hidden_node_dim = hidden_node_dim
        self.n_tasks = n_tasks
        self.task = task
        self.n_classes = n_classes
        self.num_conv = n_conv
        self.num_h = n_h
        
        # Build model architecture
        self._build_model(
            in_node_dim, hidden_node_dim, in_edge_dim,
            predictor_hidden_dim, n_conv, n_h, n_tasks, n_classes
        )

    def _validate_init_params(
        self, 
        task: str, 
        n_conv: int, 
        n_h: int, 
        n_tasks: int, 
        n_classes: int,
        in_node_dim: int, 
        hidden_node_dim: int, 
        in_edge_dim: int, 
        predictor_hidden_dim: int
    ) -> None:
        """Validate initialization parameters."""
        if task not in SUPPORTED_TASKS:
            raise ValueError(f"task must be one of {SUPPORTED_TASKS}, got '{task}'")
        
        positive_params = {
            'n_conv': n_conv, 'n_h': n_h, 'n_tasks': n_tasks,
            'in_node_dim': in_node_dim, 'hidden_node_dim': hidden_node_dim,
            'in_edge_dim': in_edge_dim, 'predictor_hidden_dim': predictor_hidden_dim
        }
        
        for param_name, param_value in positive_params.items():
            if not isinstance(param_value, int) or param_value <= 0:
                raise ValueError(f"{param_name} must be a positive integer, got {param_value}")
        
        if task == "classification" and n_classes < 2:
            raise ValueError(f"n_classes must be >= 2 for classification, got {n_classes}")

    def _build_model(
        self,
        in_node_dim: int,
        hidden_node_dim: int,
        in_edge_dim: int,
        predictor_hidden_dim: int,
        n_conv: int,
        n_h: int,
        n_tasks: int,
        n_classes: int,
    ) -> None:
        """Build the model architecture."""
        # Node embedding layer
        self.embedding = nn.Linear(in_node_dim, hidden_node_dim)
        
        # Graph convolutional layers
        self.conv_layers = nn.ModuleList([
            CGNETLayer(
                hidden_node_dim=hidden_node_dim,
                edge_dim=in_edge_dim,
                batch_norm=True,
            )
            for _ in range(n_conv)
        ])

        # MLP predictor layers
        self.fcs = self._build_mlp_predictor(hidden_node_dim, predictor_hidden_dim, n_h)
        
        # Output layer
        self.out = self._build_output_layer(predictor_hidden_dim, n_tasks, n_classes)

    def _build_mlp_predictor(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        n_layers: int
    ) -> nn.Sequential:
        """Build MLP predictor layers."""
        layers = []
        in_dim = input_dim
        
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.Softplus()
            ])
            in_dim = hidden_dim
            
        return nn.Sequential(*layers)

    def _build_output_layer(
        self, 
        input_dim: int, 
        n_tasks: int, 
        n_classes: int
    ) -> nn.Linear:
        """Build output layer based on task type."""
        if self.task == "regression":
            return nn.Linear(input_dim, n_tasks)
        else:  # classification
            return nn.Linear(input_dim, n_tasks * n_classes)

    def _compute_weighted_pool(
        self, 
        x: torch.Tensor, 
        node_weights: torch.Tensor, 
        batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute weighted global pooling.
        
        Parameters
        ----------
        x : torch.Tensor
            Node features with shape [num_nodes, num_features].
        node_weights : torch.Tensor
            Node weights with shape [num_nodes].
        batch : torch.Tensor
            Batch indices with shape [num_nodes].
            
        Returns
        -------
        torch.Tensor
            Graph-level features with shape [batch_size, num_features].
        """
        # Handle empty batch case
        if batch.numel() == 0:
            return torch.empty(0, x.size(1), device=x.device, dtype=x.dtype)
        
        batch_size = int(batch.max().item()) + 1
        
        # Compute weight sums per graph
        weight_sums = torch.zeros(
            batch_size, device=node_weights.device, dtype=node_weights.dtype
        )
        weight_sums.scatter_add_(0, batch, node_weights)
        
        # Normalize weights
        normalized_weights = node_weights / weight_sums[batch]
        
        # Apply weights and pool
        weighted_x = x * normalized_weights.unsqueeze(1)
        return global_add_pool(weighted_x, batch, size=batch_size)

    def forward(self, data: Union[Data, Batch]) -> torch.Tensor:
        """
        Forward pass of the CG-NET model.

        Parameters
        ----------
        data : torch_geometric.data.Data or torch_geometric.data.Batch
            PyG Data object containing node features (x) and edge features (edge_attr).

        Returns
        -------
        torch.Tensor
            Model outputs:
            - Regression: shape (batch_size, n_tasks) or (batch_size,) if n_tasks=1
            - Classification: 
                - Single task: (batch_size, n_classes) with logits
                - Multi task: (batch_size, n_tasks, n_classes) with logits
        """
        # Node embedding
        x = self.embedding(data.x)

        # Graph convolution layers
        for conv in self.conv_layers:
            x = conv(x, data.edge_index, data.edge_attr)

        # Get node weights (use uniform weights if not provided)
        node_weights = getattr(data, 'node_weights', None)
        if node_weights is None:
            node_weights = torch.ones(x.size(0), device=x.device, dtype=x.dtype)

        # Graph-level pooling
        g_x = F.softplus(self._compute_weighted_pool(x, node_weights, data.batch))
            
        # MLP predictor
        g_x = self.fcs(g_x)
        out = self.out(g_x)

        # Format output based on task type
        return self._format_output(out)

    def _format_output(self, out: torch.Tensor) -> torch.Tensor:
        """Format output tensor based on task type and number of tasks."""
        if self.task == "regression":
            return out.squeeze(-1) if self.n_tasks == 1 else out
        else:  # classification
            if self.n_tasks == 1:
                return out  # Shape: (batch_size, n_classes)
            else:
                # Multi-task classification: reshape to separate tasks
                batch_size = out.size(0)
                return out.view(batch_size, self.n_tasks, self.n_classes)

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the model architecture.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing model configuration and parameter counts.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'CGNETModel',
            'task': self.task,
            'n_tasks': self.n_tasks,
            'n_classes': self.n_classes if self.task == 'classification' else None,
            'hidden_node_dim': self.hidden_node_dim,
            'num_conv_layers': self.num_conv,
            'num_mlp_layers': self.num_h,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
        }


class CGNET(pl.LightningModule):
    """
    PyTorch Lightning wrapper for CG-NET model with training, validation, and testing logic.
    
    This class handles the training loop, optimization, metrics logging, and other
    training-related functionality while using CGNETModel as the core neural network.
    """
    
    def __init__(
        self,
        in_node_dim: int = DEFAULT_IN_NODE_DIM,
        hidden_node_dim: int = DEFAULT_HIDDEN_NODE_DIM,
        in_edge_dim: int = DEFAULT_IN_EDGE_DIM,
        predictor_hidden_dim: int = DEFAULT_PREDICTOR_HIDDEN_DIM,
        n_conv: int = DEFAULT_N_CONV,
        n_h: int = DEFAULT_N_H,
        n_tasks: int = DEFAULT_N_TASKS,
        task: str = "regression",
        n_classes: int = DEFAULT_N_CLASSES,
        lr: float = DEFAULT_LR,
        tmax: int = DEFAULT_TMAX,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the PyTorch Lightning CG-NET wrapper.
        
        Parameters
        ----------
        in_node_dim : int, default=92
            Initial node feature vector dimension.
        hidden_node_dim : int, default=64
            Hidden node feature vector dimension.
        in_edge_dim : int, default=41
            Initial edge feature vector dimension.
        n_conv : int, default=3
            Number of convolutional layers.
        n_h : int, default=2
            Number of hidden layers in the MLP predictor.
        predictor_hidden_dim : int, default=128
            Hidden layer size in the output MLP predictor.
        n_tasks : int, default=1
            Number of output tasks.
        task : str, default='regression'
            Task type: 'classification' or 'regression'.
        n_classes : int, default=2
            Number of classes (only used in classification mode).
        lr : float, default=1e-3
            Initial learning rate.
        tmax : int, default=10
            Number of epochs to reach minimum learning rate in cosine annealing.
        **kwargs
            Additional arguments passed to the model.
        """
        super(CGNET, self).__init__()
        self.save_hyperparameters()
        
        # Training parameters
        self.lr = lr
        self.tmax = tmax
        
        # Create the core model
        self.model = CGNETModel(
            in_node_dim=in_node_dim,
            hidden_node_dim=hidden_node_dim,
            in_edge_dim=in_edge_dim,
            predictor_hidden_dim=predictor_hidden_dim,
            n_conv=n_conv,
            n_h=n_h,
            n_tasks=n_tasks,
            task=task,
            n_classes=n_classes,
            **kwargs
        )
        
        # Store task info for logging
        self.task = task
        self.n_tasks = n_tasks
        self.n_classes = n_classes
        
        # Storage for results
        self._init_result_storage()

    def _init_result_storage(self) -> None:
        """Initialize storage for test and prediction results."""
        self.test_ids: List[Any] = []
        self.test_labels: List[torch.Tensor] = []
        self.test_outputs: List[torch.Tensor] = []
        self.predic_ids: List[Any] = []
        self.predic_labels: List[torch.Tensor] = []
        self.predic_outputs: List[torch.Tensor] = []

    def forward(self, data: Union[Data, Batch]) -> torch.Tensor:
        """Forward pass using the core model."""
        return self.model(data)

    def _compute_regression_loss_and_metrics(
        self, 
        outputs: torch.Tensor, 
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute regression loss and metrics."""
        labels = labels.view_as(outputs)
        loss = F.mse_loss(outputs, labels)
        mae = F.l1_loss(outputs, labels)
        return loss, mae

    def _compute_classification_loss_and_metrics(
        self, 
        outputs: torch.Tensor, 
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute classification loss and metrics."""
        labels = labels.long()
        
        if self.n_tasks == 1:
            # Single task classification
            loss = F.cross_entropy(outputs, labels)
            predictions = torch.argmax(outputs, dim=1)
            accuracy = (predictions == labels).float().mean()
            return loss, accuracy
        else:
            # Multi-task classification
            outputs_reshaped = outputs.view(-1, self.n_tasks, self.n_classes)
            labels_reshaped = labels.view(-1, self.n_tasks)
            
            total_loss = torch.tensor(0.0, device=outputs.device)
            total_accuracy = torch.tensor(0.0, device=outputs.device)
            
            for task_idx in range(self.n_tasks):
                task_outputs = outputs_reshaped[:, task_idx, :]
                task_labels = labels_reshaped[:, task_idx]
                task_loss = F.cross_entropy(task_outputs, task_labels)
                total_loss += task_loss
                
                # Calculate accuracy for this task
                task_predictions = torch.argmax(task_outputs, dim=1)
                task_accuracy = (task_predictions == task_labels).float().mean()
                total_accuracy += task_accuracy
            
            return total_loss / self.n_tasks, total_accuracy / self.n_tasks

    def _log_metrics(
        self, 
        loss: torch.Tensor, 
        metric: torch.Tensor, 
        batch_size: int, 
        stage: str
    ) -> None:
        """Log metrics for a given stage."""
        metric_name = "mae" if self.task == "regression" else "acc"
        
        self.log(
            f"{stage}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
        )
        self.log(
            f"{stage}_{metric_name}",
            metric,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
        )

    def training_step(self, batch: Union[Data, Batch], batch_idx: int) -> torch.Tensor:
        """Training step."""
        outputs = self.forward(batch)
        batch_size = batch.y.size(0)
        
        if self.task == "regression":
            loss, mae = self._compute_regression_loss_and_metrics(outputs, batch.y)
            self._log_metrics(loss, mae, batch_size, "train")
        else:  # classification
            loss, accuracy = self._compute_classification_loss_and_metrics(outputs, batch.y)
            self._log_metrics(loss, accuracy, batch_size, "train")
            
        return loss

    def validation_step(self, batch: Union[Data, Batch], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        outputs = self.forward(batch)
        batch_size = batch.y.size(0)
        
        if self.task == "regression":
            loss, mae = self._compute_regression_loss_and_metrics(outputs, batch.y)
            self._log_metrics(loss, mae, batch_size, "val")
        else:  # classification
            loss, accuracy = self._compute_classification_loss_and_metrics(outputs, batch.y)
            self._log_metrics(loss, accuracy, batch_size, "val")
            
        return loss

    def test_step(self, batch: Union[Data, Batch], batch_idx: int) -> torch.Tensor:
        """Test step with detailed metrics logging."""
        outputs = self.forward(batch)
        batch_size = batch.y.size(0)
        
        if self.task == "regression":
            loss, mae = self._compute_regression_loss_and_metrics(outputs, batch.y)
            self.log("test_loss", loss, on_step=False, on_epoch=True, 
                    prog_bar=False, logger=True, batch_size=batch_size)
            self.log("test_mae", mae, on_step=False, on_epoch=True, 
                    prog_bar=False, logger=True, batch_size=batch_size)
            labels = batch.y.view_as(outputs)
        else:  # classification
            loss, accuracy = self._compute_classification_loss_and_metrics(outputs, batch.y)
            self.log("test_loss", loss, on_step=False, on_epoch=True, 
                    prog_bar=False, logger=True, batch_size=batch_size)
            self.log("test_acc", accuracy, on_step=False, on_epoch=True, 
                    prog_bar=False, logger=True, batch_size=batch_size)
            
            # Additional metrics for single-task classification
            if self.n_tasks == 1:
                self._log_detailed_classification_metrics(outputs, batch.y, batch_size)
            
            labels = batch.y.long()
        
        # Store results for later analysis
        ids = self._extract_ids(batch, len(outputs))
        self.test_ids.extend(ids)
        self.test_labels.extend(labels.detach().cpu())
        self.test_outputs.extend(outputs.detach().cpu())
        
        return loss

    def _log_detailed_classification_metrics(
        self, 
        outputs: torch.Tensor, 
        labels: torch.Tensor, 
        batch_size: int
    ) -> None:
        """Log detailed classification metrics."""
        predictions = torch.argmax(outputs, dim=1)
        
        # Convert to CPU for sklearn metrics
        preds_cpu = predictions.cpu().numpy()
        labels_cpu = labels.cpu().numpy()
        
        try:
            f1 = f1_score(labels_cpu, preds_cpu, average='weighted', zero_division=0)
            precision = precision_score(labels_cpu, preds_cpu, average='weighted', zero_division=0)
            recall = recall_score(labels_cpu, preds_cpu, average='weighted', zero_division=0)
            
            self.log("test_f1", f1, on_step=False, on_epoch=True, 
                    prog_bar=False, logger=True, batch_size=batch_size)
            self.log("test_precision", precision, on_step=False, on_epoch=True, 
                    prog_bar=False, logger=True, batch_size=batch_size)
            self.log("test_recall", recall, on_step=False, on_epoch=True, 
                    prog_bar=False, logger=True, batch_size=batch_size)
        except Exception as e:
            # Log warning if metric computation fails
            print(f"Warning: Failed to compute detailed classification metrics: {e}")

    def predict_step(self, batch: Union[Data, Batch], batch_idx: int) -> None:
        """Prediction step."""
        outputs = self.forward(batch)
        
        # Extract IDs and labels
        ids = self._extract_ids(batch, len(outputs))
        labels = self._extract_labels(batch, len(outputs))
        
        # Store results
        self.predic_ids.extend(ids)
        self.predic_labels.extend(labels.detach().cpu())
        self.predic_outputs.extend(outputs.detach().cpu())

    def _extract_ids(self, batch: Union[Data, Batch], output_length: int) -> List[Any]:
        """Extract IDs from batch, using indices if not available."""
        if hasattr(batch, 'id'):
            return batch.id
        else:
            return list(range(output_length))

    def _extract_labels(self, batch: Union[Data, Batch], output_length: int) -> torch.Tensor:
        """Extract labels from batch, creating dummy labels if not available."""
        if hasattr(batch, 'y'):
            return batch.y
        else:
            # Create dummy labels with correct shape
            if self.task == "regression":
                if self.n_tasks == 1:
                    return torch.zeros(output_length)
                else:
                    return torch.zeros(output_length, self.n_tasks)
            else:  # classification
                if self.n_tasks == 1:
                    return torch.zeros(output_length, dtype=torch.long)
                else:
                    return torch.zeros(output_length, self.n_tasks, dtype=torch.long)

    def _prepare_results_dataframe(
        self, 
        ids: List[Any], 
        labels: List[torch.Tensor], 
        outputs: List[torch.Tensor]
    ) -> pd.DataFrame:
        """Prepare results dataframe based on task type."""
        # Convert tensors to numpy
        labels_cpu = [self._tensor_to_numpy(label) for label in labels]
        outputs_cpu = [self._tensor_to_numpy(output) for output in outputs]

        if self.task == "classification" and self.n_tasks == 1:
            # Single task classification with predictions and probabilities
            predictions = [np.argmax(output) if output.ndim > 0 else output for output in outputs_cpu]
            probabilities = [
                F.softmax(torch.tensor(output), dim=-1).numpy() 
                if isinstance(output, np.ndarray) and output.ndim > 0 
                else output 
                for output in outputs_cpu
            ]
            return pd.DataFrame({
                "id": ids,
                "true_label": labels_cpu,
                "predicted_label": predictions,
                "probabilities": probabilities,
                "raw_output": outputs_cpu,
            })
        elif self.task == "classification":
            # Multi-task classification
            return pd.DataFrame({
                "id": ids,
                "true_label": labels_cpu,
                "raw_output": outputs_cpu,
            })
        else:
            # Regression task
            return pd.DataFrame({
                "id": ids,
                "label": labels_cpu,
                "output": outputs_cpu,
            })

    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy array safely."""
        if torch.is_tensor(tensor):
            return tensor.cpu().numpy()
        else:
            return tensor

    def _save_results(self, df: pd.DataFrame, filename: str) -> None:
        """Save results dataframe to CSV."""
        if hasattr(self.logger, 'log_dir') and self.logger.log_dir:
            save_path = os.path.join(self.logger.log_dir, filename)
        else:
            save_path = filename
        
        df.to_csv(save_path, index=False)
        print(f"Results saved to {save_path}")

    def on_test_epoch_end(self) -> None:
        """Process and save test results at epoch end."""
        df = self._prepare_results_dataframe(self.test_ids, self.test_labels, self.test_outputs)
        self._save_results(df, "test_results.csv")
        
        # Clear stored results
        self.test_ids.clear()
        self.test_labels.clear()
        self.test_outputs.clear()

    def on_predict_epoch_end(self) -> None:
        """Process and save prediction results at epoch end."""
        df = self._prepare_results_dataframe(self.predic_ids, self.predic_labels, self.predic_outputs)
        self._save_results(df, "predict_results.csv")
        
        # Clear stored results
        self.predic_ids.clear()
        self.predic_labels.clear()
        self.predic_outputs.clear()

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler._LRScheduler]]:
        """Configure optimizers and learning rate schedulers."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.tmax
        )
        return [optimizer], [scheduler]

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model architecture."""
        return self.model.get_model_info()
