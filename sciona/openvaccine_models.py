"""Hash-verified OpenVaccine models for the pending Community execution.

Caller provisions TensorFlow with legacy Keras before importing this module's
factory. No source dataset I/O, weight download, or implicit seed reset.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

SOURCE_SHA256 = '01f27b6401bceba9ad6742f680d799cea552d918adf5e322474f04f90223607e'
LICENSE_SHA256 = '2cf381fc614bb01102e06f2e08b75c1d6a4d908426b306d18614089e044ae206'
FAMILIES = ('autoencoder', 'gru', 'lstm', 'forward', 'wave')
_DEFINITIONS = {'MSE','get_base','get_ae_model','get_model','forward','res','attention',
                'multi_head_attention','adj_attn','gru_layer','lstm_layer','wave_block','get_optimizer'}


def validate_inputs(nodes, adjacency):
    nodes, adjacency = np.asarray(nodes), np.asarray(adjacency)
    if nodes.dtype != np.float32 or adjacency.dtype != np.float32:
        raise ValueError('Model inputs must be float32')
    if nodes.ndim != 3 or nodes.shape[0] < 1 or nodes.shape[1] < 4 or nodes.shape[2] != 55:
        raise ValueError('Expected nonempty node batches with boundary tokens and 55 features')
    if adjacency.shape != (nodes.shape[0], nodes.shape[1], nodes.shape[1], 8):
        raise ValueError('Adjacency must match node batch and length with 8 channels')
    if not np.isfinite(nodes).all() or not np.isfinite(adjacency).all():
        raise ValueError('Model inputs must be finite')
    return nodes, adjacency


@dataclass
class OpenVaccineModel:
    family: str
    base: Any
    model: Any
    tf: Any

    def predict(self, nodes, adjacency):
        nodes, adjacency = validate_inputs(nodes, adjacency)
        if self.family == 'autoencoder':
            raise ValueError('Autoencoder returns reconstruction loss, not predictions')
        output = self.model([nodes, adjacency], training=False).numpy()
        if output.shape != (nodes.shape[0], nodes.shape[1]-2, 5) or not np.isfinite(output).all():
            raise ValueError('Invalid model output')
        return output

    def train_step(self, nodes, adjacency, targets=None, sample_weights=None):
        nodes, adjacency = validate_inputs(nodes, adjacency)
        tf = self.tf
        if self.family == 'autoencoder':
            if targets is not None or sample_weights is not None:
                raise ValueError('Autoencoder reconstructs its inputs without targets or weights')
        else:
            targets = np.asarray(targets)
            sample_weights = np.asarray(sample_weights)
            if targets.dtype != np.float32 or targets.shape != (nodes.shape[0], nodes.shape[1]-2, 5):
                raise ValueError('Targets must be float32 and match model output')
            if np.isinf(targets).any() or not np.isfinite(targets).any():
                raise ValueError('Targets need observed finite values; NaN denotes masked values')
            if (sample_weights.dtype != np.float32 or sample_weights.shape != (nodes.shape[0],)
                    or not np.isfinite(sample_weights).all() or (sample_weights < 0).any()
                    or not np.isfinite(sample_weights.sum()) or sample_weights.sum() <= 0):
                raise ValueError('Explicit finite nonnegative sample weights with positive sum required')
        with tf.GradientTape() as tape:
            output = self.model([nodes, adjacency], training=True)
            loss = (tf.reduce_mean(output) if self.family == 'autoencoder'
                    else self.model.loss(tf.constant(targets), output, tf.constant(sample_weights)))
        gradients = tape.gradient(loss, self.model.trainable_variables)
        if not np.isfinite(loss.numpy()).all() or any(g is None or not np.isfinite(g.numpy()).all() for g in gradients):
            raise ValueError('Nonfinite training loss or gradient')
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return float(loss.numpy())


def create_model(source_dir: str | Path, family: str, *, base_weights=None):
    if family not in FAMILIES:
        raise ValueError('Unsupported OpenVaccine model family')
    source_dir = Path(source_dir)
    raw = (source_dir/'scripts/nullrecurrent_inference.py').read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError('OpenVaccine source identity mismatch')
    if hashlib.sha256((source_dir/'LICENSE').read_bytes()).hexdigest() != LICENSE_SHA256:
        raise ValueError('OpenVaccine license identity mismatch')
    import tensorflow as tf
    if tf.keras.layers.Layer.__module__.split('.')[0] != 'tf_keras':
        raise RuntimeError('OpenVaccine currently requires explicitly provisioned legacy Keras')
    definitions = [n for n in ast.parse(raw).body
                   if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in _DEFINITIONS]
    if {n.name for n in definitions} != _DEFINITIONS or len(definitions) != len(_DEFINITIONS):
        raise ValueError('Incomplete source model definitions')
    nodes = np.zeros((1,4,55), dtype=np.float32)
    adjacency = np.zeros((1,4,4,8), dtype=np.float32)
    ns = dict(np=np, tf=tf, L=tf.keras.layers, K=tf.keras.backend, losses=tf.keras.losses,
              LOSS_WGTS=[.3,.3,.3,.05,.05], X_node=nodes, As=adjacency)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), '<pinned-openvaccine-models>', 'exec'), ns)
    base = ns['get_base']({},nodes,adjacency)
    if base_weights is not None:
        base.set_weights(base_weights)
    model = (ns['get_ae_model'](base,{}) if family == 'autoencoder'
             else ns['get_model'](base,{},family,nodes,adjacency))
    return OpenVaccineModel(family,base,model,tf)
