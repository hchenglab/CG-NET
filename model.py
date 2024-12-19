from __future__ import annotations

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

try:
    import dgl
except:
    raise ImportError("This class requires DGL to be installed.")


class CGCNNLayer(nn.Module):
    def __init__(self, hidden_node_dim: int, edge_dim: int, batch_norm: bool = True):
        """
        Parameters
        ----------
        hidden_node_dim: int
            The length of the hidden node feature vectors.
        edge_dim: int
            The length of the edge feature vectors.
        batch_norm: bool, default True
            Whether to apply batch normalization or not.
        """
        super(CGCNNLayer, self).__init__()
        z_dim = 2 * hidden_node_dim + edge_dim
        liner_out_dim = 2 * hidden_node_dim
        self.linear = nn.Linear(z_dim, liner_out_dim)
        self.batch_norm = nn.BatchNorm1d(liner_out_dim) if batch_norm else None

    def message_func(self, edges):
        z = torch.cat([edges.src["x"], edges.dst["x"], edges.data["edge_attr"]], dim=1)
        z = self.linear(z)
        if self.batch_norm is not None:
            z = self.batch_norm(z)
        gated_z, message_z = z.chunk(2, dim=1)
        gated_z = torch.sigmoid(gated_z)
        message_z = F.softplus(message_z)
        return {"message": gated_z * message_z}

    def reduce_func(self, nodes):
        msgs = nodes.mailbox["message"]
        if msgs.size(1) == 0:
            return {"new_x": F.softplus(nodes.data["x"])}
        else:
            nbr_sumed = torch.sum(msgs, dim=1)
            new_x = F.softplus(nodes.data["x"] + nbr_sumed)
            return {"new_x": new_x}

    def forward(self, dgl_graph, node_feats, edge_feats):
        """Update node representations.

        Parameters
        ----------
        dgl_graph: DGLGraph
            DGLGraph for a batch of graphs.
        node_feats: torch.Tensor
            The node features. The shape is `(N, hidden_node_dim)`.
        edge_feats: torch.Tensor
            The edge features. The shape is `(N, hidden_node_dim)`.

        Returns
        -------
        node_feats: torch.Tensor
            The updated node features. The shape is `(N, hidden_node_dim)`.
        """
        dgl_graph.ndata["x"] = node_feats
        dgl_graph.edata["edge_attr"] = edge_feats
        dgl_graph.update_all(self.message_func, self.reduce_func)
        node_feats = dgl_graph.ndata.pop("new_x")
        return node_feats


class CGCNN(pl.LightningModule):
    def __init__(
        self,
        in_node_dim: int = 92,
        hidden_node_dim: int = 64,
        in_edge_dim: int = 41,
        predictor_hidden_feats: int = 128,
        num_conv: int = 3,
        n_tasks: int = 1,
        task: str = "regression",
        n_classes: int = 2,
        lr=1e-3,
        tmax=10,
        **kwargs,
    ):
        """
        Parameters
        ----------
        in_node_dim: int, default 92
            The length of the initial node feature vectors. The 92 is
            based on length of vectors in the atom_init.json.
        hidden_node_dim: int, default 64
            The length of the hidden node feature vectors.
        in_edge_dim: int, default 41
            The length of the initial edge feature vectors. The 41 is
            based on default setting of CGCNNFeaturizer.
        num_conv: int, default 3
            The number of convolutional layers.
        predictor_hidden_feats: int, default 128
            The size for hidden representations in the output MLP predictor.
        n_tasks: int, default 1
            The number of the output size.
        task: str, default 'regression'
            The task type, 'classification' or 'regression'.
        n_classes: int, default 2
            The number of classes to predict (only used in classification mode).
        lr: float, default 1e-3
            The initial learning rate.
        tmax: int, default 10
            The number of epochs to reach the minimum learning rate.
        """

        super(CGCNN, self).__init__()
        self.save_hyperparameters()
        if task not in ["classification", "regression"]:
            raise ValueError("mode must be either 'classification' or 'regression'")

        self.hidden_node_dim = hidden_node_dim
        self.n_tasks = n_tasks
        self.task = task
        self.n_classes = n_classes
        self.lr = lr
        self.tmax = tmax
        self.embedding = nn.Linear(in_node_dim, hidden_node_dim)
        self.conv_layers = nn.ModuleList(
            [
                CGCNNLayer(
                    hidden_node_dim=hidden_node_dim,
                    edge_dim=in_edge_dim,
                    batch_norm=True,
                )
                for _ in range(num_conv)
            ]
        )
        self.readout = dgl.mean_nodes
        self.fc = nn.Linear(hidden_node_dim, predictor_hidden_feats)
        if self.task == "regression":
            self.out = nn.Linear(predictor_hidden_feats, n_tasks)
        else:
            self.out = nn.Linear(predictor_hidden_feats, n_tasks * n_classes)
        self.test_ids = []
        self.test_labels = []
        self.test_outputs = []
        self.predic_ids = []
        self.predic_labels = []
        self.predic_outputs = []
        

    def forward(self, dgl_graph):
        """Predict labels

        Parameters
        ----------
        dgl_graph: DGLGraph
            DGLGraph for a batch of graphs. The graph expects that the node features
            are stored in `ndata['x']`, and the edge features are stored in `edata['edge_attr']`.

        Returns
        -------
        out: torch.Tensor
            The output values of this model.
            If mode == 'regression', the shape is `(batch_size, n_tasks)`.
            If mode == 'classification', the shape is `(batch_size, n_tasks, n_classes)` (n_tasks > 1)
            or `(batch_size, n_classes)` (n_tasks == 1) and the output values are probabilities of each class label.
        """
        graph = dgl_graph
        # embedding node features
        node_feats = graph.ndata.pop("x")
        edge_feats = graph.edata.pop("edge_attr")
        node_feats = self.embedding(node_feats)

        # convolutional layer
        for conv in self.conv_layers:
            node_feats = conv(graph, node_feats, edge_feats)

        # pooling
        graph.ndata["updated_x"] = node_feats
        graph_feat = F.softplus(self.readout(graph, "updated_x", "w"))
        graph_feat = F.softplus(self.fc(graph_feat))
        out = self.out(graph_feat)

        if self.task == "regression":
            return out.squeeze(-1)
        else:
            logits = out.view(-1, self.n_tasks, self.n_classes)
            # for n_tasks == 1 case
            logits = torch.squeeze(logits)
            proba = F.softmax(logits)
            return proba, logits
        
    def pooling(self, graph, feat, weight, op="mean"):
        graphs = dgl.unbatch(graph)
        weighted_h = torch.zeros(graph.batch_size, self.hidden_node_dim).to(graph.device)
        for i, g in enumerate(graphs):
            h = g.ndata[feat]
            w = g.ndata[weight]
            in_degree = g.in_degrees()
            none_zero_in_degree_mask = in_degree > 0
            h_filtered = h[none_zero_in_degree_mask]
            w_filtered = w[none_zero_in_degree_mask]
            if op == "mean":
                weighted_h[i] = torch.sum(h_filtered * w_filtered, dim=0) / torch.sum(w_filtered)
            elif op == "sum":
                weighted_h[i] = torch.sum(h_filtered * w_filtered, dim=0)
        return weighted_h

    def training_step(self, batch, batch_idx):
        ids, inputs, labels = batch
        outputs = self.forward(inputs)
        if self.task == "regression":
            labels = labels.view_as(outputs)
            loss = F.mse_loss(outputs, labels)
            mae = F.l1_loss(outputs, labels)
            self.log(
                "train_loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            self.log(
                "train_mae",
                mae,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        return loss

    def validation_step(self, batch, batch_idx):
        ids, inputs, labels = batch
        outputs = self.forward(inputs)
        if self.task == "regression":
            labels = labels.view_as(outputs)
            loss = F.mse_loss(outputs, labels)
            mae = F.l1_loss(outputs, labels)
            self.log(
                "val_loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            self.log(
                "val_mae", mae, on_step=False, on_epoch=True, prog_bar=True, logger=True
            )
        return loss

    def test_step(self, batch, batch_idx):
        ids, inputs, labels = batch
        outputs = self.forward(inputs)
        if self.task == "regression":
            labels = labels.view_as(outputs)
            loss = F.mse_loss(outputs, labels)
            mae = F.l1_loss(outputs, labels)
            self.log(
                "test_loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )
            self.log(
                "test_mae",
                mae,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )
        self.test_ids.extend(ids)
        self.test_labels.extend(labels)
        self.test_outputs.extend(outputs)

    def predict_step(self, batch, batch_idx):
        ids, inputs, labels = batch
        outputs = self.forward(inputs)
        self.predic_ids.extend(ids)
        self.predic_labels.extend(labels)
        self.predic_outputs.extend(outputs)
    
    def on_test_epoch_end(self, path='.'):
        # Move tensors to CPU
        test_ids = self.test_ids
        test_labels_cpu = [label.cpu().numpy() for label in self.test_labels]
        test_outputs_cpu = [output.cpu().numpy() for output in self.test_outputs]

        df = pd.DataFrame(
            {
                "id": test_ids,
                "label": test_labels_cpu,
                "output": test_outputs_cpu,
            }
        )
        df.to_csv(os.path.join(self.logger.log_dir, "test_results.csv"), index=False)

    def on_predict_epoch_end(self):
        # Move tensors to CPU
        predic_ids = self.predic_ids
        predic_labels_cpu = [label.cpu().numpy() for label in self.predic_labels]
        predic_outputs_cpu = [output.cpu().numpy() for output in self.predic_outputs]

        df = pd.DataFrame(
            {
                "id": predic_ids,
                "label": predic_labels_cpu,
                "output": predic_outputs_cpu,
            }
        )
        df.to_csv(os.path.join(self.logger.log_dir, "predict_results.csv"), index=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.tmax
        )
        return [optimizer], [scheduler]
