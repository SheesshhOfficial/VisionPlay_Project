"""Local supervised learning. No pretrained or synthetic SVM is shipped."""
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from core import LABELS

HEADER = ['session', 'label'] + [f'f{i}' for i in range(63)]

def add_sample(path, session, label, vector):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with path.open('a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if new: writer.writerow(HEADER)
        writer.writerow([session, label, *vector])

def read_samples(path):
    with Path(path).open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != HEADER:
            raise ValueError('Dataset header is invalid; restore or rename data/samples.csv.')
        rows = list(reader)
    x = np.asarray([[float(r[f'f{i}']) for i in range(63)] for r in rows])
    y = np.asarray([r['label'] for r in rows])
    groups = np.asarray([r['session'] for r in rows])
    if not rows or not np.isfinite(x).all() or not set(y) <= set(LABELS):
        raise ValueError('Dataset is empty or contains invalid values.')
    return x, y, groups

def estimator():
    return make_pipeline(StandardScaler(), SVC(C=3.0, kernel='rbf', gamma='scale'))

def train(path, report_path):
    if not Path(path).exists():
        raise ValueError('Collect samples first: 1, 2, 3 in each of 3 sessions (N).')
    x, y, groups = read_samples(path)
    sessions = list(dict.fromkeys(groups))
    counts = Counter(zip(groups, y))
    complete = [s for s in sessions if all(counts[s, label] >= 20 for label in LABELS)]
    if len(complete) < 3:
        raise ValueError('Need 3 complete sessions: collect 1, 2, 3; N; repeat twice.')
    # Incomplete sessions are excluded so every session represents every class.
    valid = np.isin(groups, complete)
    test = valid & (groups == complete[-1])
    fit = valid & ~test
    evaluation_model = estimator().fit(x[fit], y[fit])
    predicted = evaluation_model.predict(x[test])
    report = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'method': 'Newest complete capture session held out; no frames from it in evaluation training.',
        'classes': list(LABELS), 'train_sessions': complete[:-1], 'test_session': complete[-1],
        'train_samples': int(fit.sum()), 'test_samples': int(test.sum()),
        'excluded_incomplete_samples': int((~valid).sum()),
        'accuracy': float(accuracy_score(y[test], predicted)),
        'classification_report': classification_report(y[test], predicted, labels=LABELS, output_dict=True, zero_division=0),
        'confusion_matrix_rows_true_columns_predicted': confusion_matrix(y[test], predicted, labels=LABELS).tolist(),
        'warning': 'Small personal dataset; not general accuracy. Do not repeatedly tune against this holdout. Final live model refit on all complete sessions; its score is not independently measured.'
    }
    Path(report_path).write_text(json.dumps(report, indent=2), encoding='utf-8')
    live_model = estimator().fit(x[valid], y[valid])
    return live_model, report
