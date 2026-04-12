"""
Module 2 – Spatio-Temporal GNN Predictor
=========================================
A lightweight GCN + GRU model built on TensorFlow / Spektral.

Architecture per time-step:
    Node features → GCNConv → ReLU → GCNConv → ReLU  (spatial)
    Stacked over T steps → GRU → Dense               (temporal)

Input shapes
    X : (batch, T, N, F)   — windowed node features
    A : (N, N)              — adjacency matrix (shared, sparse-friendly)

Output
    Ŷ : (batch, N)          — predicted demand at each node for t+1

Downstream (hybrid scheduling)
    Predictions are combined with mean traffic_speed per stop and the same
    road graph in ``utils.gnn_propagation`` to diffuse congestion stress along
    edges—spatial spillover aligned with GCN message passing—then used as
    per-stop wait weights in ``optimizers.network_optimizer``.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from spektral.layers import GCNConv


class MaskSafeGCNConv(GCNConv):
    """
    Spektral GCNConv assumes `mask` is either absent or a real tensor mask.
    Keras 3 can supply `mask=(None, None)` for list inputs, which is truthy
    but breaks `output *= mask[0]`. Force-disable masking for graph tuple inputs.
    """

    def call(self, inputs, mask=None, **kwargs):
        return super().call(inputs, mask=None)


class GcnEncoderBlock(layers.Layer):
    """Two-layer GCN encoder for one time step, batch × nodes × features."""

    def __init__(self, hidden: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.gcn1 = MaskSafeGCNConv(hidden, activation="relu")
        self.gcn2 = MaskSafeGCNConv(hidden, activation="relu")

    def call(self, inputs, training=False):
        x, a = inputs
        x = self.gcn1([x, a])
        x = self.gcn2([x, a])
        return x


class SpatioTemporalGNN(keras.Model):
    """
    End-to-end Spatio-Temporal GNN.

    Parameters
    ----------
    n_nodes : int
        Number of bus stops (graph nodes).
    gcn_hidden : int
        Hidden units in each GCN layer.
    gru_hidden : int
        Hidden units in the GRU temporal encoder.
    """

    def __init__(
        self,
        n_nodes: int,
        window: int,
        gcn_hidden: int = 32,
        gru_hidden: int = 64,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_nodes = n_nodes
        self.window = window
        self.gcn_block = GcnEncoderBlock(hidden=gcn_hidden)
        self.gru = layers.GRU(gru_hidden, return_sequences=False)
        self.dense_out = layers.Dense(n_nodes)

    def call(self, inputs, training=False):
        """
        Parameters
        ----------
        inputs : tuple (x_seq, a)
            x_seq : Tensor (batch, T, N, F)
            a      : Tensor (N, N) — GCN-filtered adjacency (see spektral.utils.convolution.gcn_filter)
        """
        x_seq, a = inputs
        batch_size = tf.shape(x_seq)[0]
        # Fixed window avoids `tf.range` on symbolic T (Keras 3 graph tracing).
        seq = []
        for t in range(self.window):
            x_t = x_seq[:, t, :, :]  # (batch, N, F)
            step_out = self.gcn_block([x_t, a], training=training)  # (batch, N, H)
            seq.append(tf.reshape(step_out, [batch_size, -1]))  # (batch, N*H)

        temporal_input = tf.stack(seq, axis=1)
        gru_out = self.gru(temporal_input, training=training)
        return self.dense_out(gru_out)


# ---------------------------------------------------------------------------
# Helper: compile & build with standard settings
# ---------------------------------------------------------------------------
def build_model(
    n_nodes: int,
    window: int = 4,
    n_features: int = 3,
    gcn_hidden: int = 32,
    gru_hidden: int = 64,
    lr: float = 1e-3,
) -> SpatioTemporalGNN:
    model = SpatioTemporalGNN(
        n_nodes=n_nodes,
        window=window,
        gcn_hidden=gcn_hidden,
        gru_hidden=gru_hidden,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=["mae"],
    )
    # Trigger weight creation
    dummy_x = tf.zeros((1, window, n_nodes, n_features))
    dummy_a = tf.eye(n_nodes)
    model((dummy_x, dummy_a))
    return model
