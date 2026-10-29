"""Trainable neural classifier for VisionPlay's 63 landmark features.

This is a multi-layer perceptron over hand landmarks, not a pixel CNN. Model
artifacts use numeric NPZ arrays only; loading never enables pickle.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from core import LABELS
from learning import HEADER

FEATURES = 63
HIDDEN_LAYERS = (128, 64, 32)
SEED = 1729
MAX_ITER = 160
SCHEMA = 1


def _dataset(path):
    path = Path(path)
    if not path.exists():
        raise ValueError('No landmark dataset found. Open Guided Training and capture all 3 poses in 3 sessions first.')
    raw = path.read_bytes()
    try:
        with path.open(newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != HEADER:
                raise ValueError('Dataset header is invalid; expected session,label,f0...f62.')
            rows = list(reader)
        if not rows:
            raise ValueError('Dataset is empty.')
        x = np.asarray([[float(row[f'f{i}']) for i in range(FEATURES)] for row in rows], dtype=np.float64)
        y = np.asarray([row['label'] for row in rows], dtype=str)
        groups = np.asarray([row['session'] for row in rows], dtype=str)
    except (KeyError, TypeError, ValueError, csv.Error) as exc:
        raise ValueError(f'Dataset contains invalid rows: {exc}') from exc
    if x.ndim != 2 or x.shape[1] != FEATURES or not np.isfinite(x).all():
        raise ValueError('Dataset must contain 63 finite landmark features per row.')
    if not set(y) <= set(LABELS) or not all(y) or not all(groups):
        raise ValueError('Dataset contains unknown labels or empty session names.')
    return x, y, groups, hashlib.sha256(raw).hexdigest()


def _partition(y, groups):
    sessions = list(dict.fromkeys(groups.tolist()))
    counts = Counter(zip(groups.tolist(), y.tolist()))
    complete = [s for s in sessions if all(counts[s, label] >= 20 for label in LABELS)]
    if len(complete) < 3:
        raise ValueError('Need at least 3 complete sessions with at least 20 samples per class each. Use Capture / Next in the training wizard.')
    eligible = np.isin(groups, complete)
    test = eligible & (groups == complete[-1])
    fit = eligible & ~test
    # Defend against datasets with odd labeling or class omissions.
    if set(y[fit]) != set(LABELS) or set(y[test]) != set(LABELS):
        raise ValueError('Training and newest-session holdout must each contain all three classes.')
    return complete, fit, test


class LandmarkMLP:
    """Validated inference wrapper compatible with predict / predict_proba."""
    def __init__(self, scaler_mean, scaler_scale, layers, biases, classes=LABELS):
        self.mean = np.asarray(scaler_mean, dtype=np.float64)
        self.scale = np.asarray(scaler_scale, dtype=np.float64)
        self.weights = [np.asarray(w, dtype=np.float64) for w in layers]
        self.biases = [np.asarray(b, dtype=np.float64) for b in biases]
        self.classes_ = np.asarray(classes, dtype=str)
        self._validate()

    def _validate(self):
        if self.mean.shape != (FEATURES,) or self.scale.shape != (FEATURES,) or len(self.weights) != 4 or len(self.biases) != 4:
            raise ValueError('Model has invalid feature or layer shapes.')
        shapes = [(63, 128), (128, 64), (64, 32), (32, 3)]
        for w, b, shape, n in zip(self.weights, self.biases, shapes, (128, 64, 32, 3)):
            if w.shape != shape or b.shape != (n,):
                raise ValueError('Model has invalid network parameter shapes.')
        arrays = [self.mean, self.scale, *self.weights, *self.biases]
        if not all(np.isfinite(a).all() for a in arrays) or np.any(self.scale <= 0):
            raise ValueError('Model parameters must be finite and scaler scales positive.')
        if tuple(self.classes_) != tuple(LABELS):
            raise ValueError('Model class order is invalid.')

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != FEATURES or not np.isfinite(X).all():
            raise ValueError('Expected a finite 2-D array with 63 landmark features per row.')
        a = (X - self.mean) / self.scale
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            a = np.maximum(a @ w + b, 0.0)
        logits = a @ self.weights[-1] + self.biases[-1]
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs = probs / probs.sum(axis=1, keepdims=True)
        # MLPClassifier's output columns follow sorted clf.classes_ order.
        # Reorder to the public LABELS contract (rock, paper, scissors).
        return probs[:, [tuple(sorted(LABELS)).index(c) for c in LABELS]]

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def _package(scaler, clf):
    return LandmarkMLP(scaler.mean_, scaler.scale_, clf.coefs_, clf.intercepts_)


def _atomic_npz(path, model, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {'mean': model.mean, 'scale': model.scale, 'metadata': np.asarray(json.dumps(metadata, separators=(',', ':')))}
    for i, (w, b) in enumerate(zip(model.weights, model.biases)):
        arrays[f'w{i}'] = w
        arrays[f'b{i}'] = b
    fd, temp = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            np.savez_compressed(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    except Exception:
        try: os.unlink(temp)
        except OSError: pass
        raise


def load_neural(model_path):
    """Load numeric-only artifact, validating schema, shapes and finite values."""
    try:
        with np.load(model_path, allow_pickle=False) as z:
            required = {'mean', 'scale', 'metadata'} | {f'{p}{i}' for i in range(4) for p in ('w', 'b')}
            if set(z.files) != required:
                raise ValueError('Model archive keys do not match expected schema.')
            metadata = json.loads(str(z['metadata'].item()))
            if not isinstance(metadata, dict) or metadata.get('schema') != SCHEMA or metadata.get('feature_count') != FEATURES:
                raise ValueError('Unsupported neural model metadata schema.')
            model = LandmarkMLP(z['mean'], z['scale'], [z[f'w{i}'] for i in range(4)], [z[f'b{i}'] for i in range(4)], metadata.get('classes', ()))
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f'Cannot load neural model: {exc}') from exc
    model.metadata = metadata
    return model


def train_neural(data_path, model_path, report_path):
    """Fit once on prior complete sessions, evaluate newest, save that exact model."""
    x, y, groups, fingerprint = _dataset(data_path)
    complete, fit, test = _partition(y, groups)
    scaler = StandardScaler().fit(x[fit])
    clf = MLPClassifier(hidden_layer_sizes=HIDDEN_LAYERS, activation='relu', solver='adam',
                        random_state=SEED, max_iter=MAX_ITER, early_stopping=False,
                        tol=1e-4, n_iter_no_change=20)
    clf.fit(scaler.transform(x[fit]), y[fit])
    model = _package(scaler, clf)
    predicted = model.predict(x[test])
    report = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'model': 'StandardScaler + MLPClassifier landmark MLP (hidden layers 128,64,32; ReLU; Adam)',
        'input': '63 normalized xyz hand-landmark features; not raw pixels and not a CNN',
        'seed': SEED, 'max_iter': MAX_ITER, 'iterations': int(clf.n_iter_),
        'train_sessions': complete[:-1], 'test_session': complete[-1],
        'train_samples': int(fit.sum()), 'test_samples': int(test.sum()),
        'excluded_incomplete_samples': int((~np.isin(groups, complete)).sum()),
        'accuracy': float(accuracy_score(y[test], predicted)),
        'classification_report': classification_report(y[test], predicted, labels=LABELS, output_dict=True, zero_division=0),
        'confusion_matrix_rows_true_columns_predicted': confusion_matrix(y[test], predicted, labels=LABELS).tolist(),
        'training_loss_curve': [float(v) for v in clf.loss_curve_],
        'dataset_sha256': fingerprint,
        'limitations': 'Small personal landmark dataset; this held-out session score is not general accuracy. Do not repeatedly tune against this holdout. The exported model is the exact model evaluated; no all-data refit. Synthetic tests are not real gesture accuracy measurements.'
    }
    metadata = {'schema': SCHEMA, 'feature_count': FEATURES, 'classes': list(LABELS),
                'dataset_sha256': fingerprint, 'trained_utc': report['created_utc'],
                'test_session': complete[-1], 'train_sessions': complete[:-1],
                'evaluation': report}
    # The model archive is authoritative and is written atomically as one artifact.
    # The sidecar report is a convenience copy; if it fails, retain the valid model
    # and explicitly report that the embedded evaluation is available in metadata.
    _atomic_npz(model_path, model, metadata)
    rp = Path(report_path)
    temp = None
    try:
        rp.parent.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(prefix=f'.{rp.name}.', suffix='.tmp', dir=rp.parent)
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
            f.flush(); os.fsync(f.fileno())
        os.replace(temp, rp)
        temp = None
    except Exception as exc:
        if temp is not None:
            try: os.unlink(temp)
            except OSError: pass
        report['report_sidecar_error'] = f'{type(exc).__name__}: {exc}; authoritative report remains embedded in model artifact metadata.'
    model.metadata = metadata
    return model, report
