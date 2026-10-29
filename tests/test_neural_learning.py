"""Synthetic fixtures test software behavior only, not real gesture accuracy."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import LABELS
from learning import HEADER, add_sample
from neural_learning import train_neural, load_neural
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def fixture(path, sessions=('s1', 's2', 's3'), per_class=20, seed=20):
    rng = np.random.default_rng(seed)
    # Separated synthetic blobs ensure the optimization remains quick and stable.
    for si, session in enumerate(sessions):
        for ci, label in enumerate(LABELS):
            for _ in range(per_class):
                add_sample(path, session, label, rng.normal(ci * 2, .12 + si * .01, 63))


class NeuralLearningTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data, self.model, self.report = (self.root / 'samples.csv', self.root / 'model.npz', self.root / 'report.json')

    def tearDown(self): self.tmp.cleanup()

    def test_split_no_leakage_and_report(self):
        fixture(self.data)
        model, report = train_neural(self.data, self.model, self.report)
        self.assertEqual(report['train_sessions'], ['s1', 's2'])
        self.assertEqual(report['test_session'], 's3')
        self.assertEqual(report['train_samples'], 120)
        self.assertEqual(report['test_samples'], 60)
        self.assertIn('not raw pixels', report['input'])
        self.assertEqual(len(report['confusion_matrix_rows_true_columns_predicted']), 3)
        self.assertGreater(len(report['training_loss_curve']), 0)
        self.assertTrue(self.model.exists() and self.report.exists())

    def test_reproducibility(self):
        fixture(self.data)
        m1, r1 = train_neural(self.data, self.model, self.report)
        a = m1.predict_proba(np.zeros((5, 63)))
        m2, r2 = train_neural(self.data, self.root / 'm2.npz', self.root / 'r2.json')
        np.testing.assert_allclose(a, m2.predict_proba(np.zeros((5, 63))), atol=1e-12)
        self.assertEqual(r1['accuracy'], r2['accuracy'])

    def test_persistence_probability_parity_and_stale_fingerprint(self):
        fixture(self.data)
        model, report = train_neural(self.data, self.model, self.report)
        loaded = load_neural(self.model)
        x = np.random.default_rng(3).normal(size=(4, 63))
        np.testing.assert_allclose(model.predict_proba(x), loaded.predict_proba(x), atol=1e-12)
        np.testing.assert_array_equal(model.predict(x), loaded.predict(x))
        self.assertEqual(loaded.metadata['dataset_sha256'], report['dataset_sha256'])
        self.assertIn('evaluation', loaded.metadata)
        self.assertEqual(loaded.metadata['evaluation']['accuracy'], report['accuracy'])
        # Explicitly compare canonical labels/probabilities to sklearn's sorted
        # class order; sklearn sorts classes as paper, rock, scissors.
        with self.data.open(newline='') as f:
            rows = list(csv.DictReader(f))
        train_rows = [r for r in rows if r['session'] in ('s1', 's2')]
        Xtr = np.asarray([[float(r[f'f{i}']) for i in range(63)] for r in train_rows])
        ytr = np.asarray([r['label'] for r in train_rows])
        scaler = StandardScaler().fit(Xtr)
        reference = MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam',
                                  random_state=1729, max_iter=160, early_stopping=False,
                                  tol=1e-4, n_iter_no_change=20).fit(scaler.transform(Xtr), ytr)
        np.testing.assert_array_equal(loaded.classes_, LABELS)
        np.testing.assert_allclose(loaded.predict_proba(x),
                                   reference.predict_proba(scaler.transform(x))[:, [1, 0, 2]], atol=1e-12)
        np.testing.assert_array_equal(loaded.predict(x), reference.predict(scaler.transform(x)))
        self.assertFalse(np.load(self.model, allow_pickle=False)['metadata'].dtype.hasobject)

    def test_rejected_bad_data_variants(self):
        fixture(self.data, per_class=19)
        with self.assertRaisesRegex(ValueError, '3 complete sessions'):
            train_neural(self.data, self.model, self.report)
        with self.data.open('w', newline='') as f: csv.writer(f).writerow(['bad'])
        with self.assertRaisesRegex(ValueError, 'header'):
            train_neural(self.data, self.model, self.report)
        self.data.write_text(','.join(HEADER) + '\n')
        with self.assertRaisesRegex(ValueError, 'empty'):
            train_neural(self.data, self.model, self.report)

    def test_nonfinite_csv_and_invalid_predict_inputs(self):
        fixture(self.data)
        # Inject a NaN while retaining otherwise valid schema and enough samples.
        with self.data.open() as f:
            rows = list(csv.reader(f))
        rows[1][2] = 'nan'
        with self.data.open('w', newline='') as f: csv.writer(f).writerows(rows)
        with self.assertRaisesRegex(ValueError, 'finite'):
            train_neural(self.data, self.model, self.report)
        self.data.unlink()
        fixture(self.data, seed=22)
        model, _ = train_neural(self.data, self.model, self.report)
        for bad in (np.zeros(63), np.zeros((1, 62)), np.full((1, 63), np.inf)):
            with self.assertRaises(ValueError): model.predict_proba(bad)

    def test_missing_class_and_missing_data(self):
        # Every session lacks scissors.
        rng = np.random.default_rng(1)
        for s in ('a', 'b', 'c'):
            for label in LABELS[:2]:
                for _ in range(25): add_sample(self.data, s, label, rng.normal(size=63))
        with self.assertRaisesRegex(ValueError, '3 complete sessions'):
            train_neural(self.data, self.model, self.report)
        with self.assertRaisesRegex(ValueError, 'No landmark dataset'):
            train_neural(self.root / 'absent.csv', self.model, self.report)

    def test_invalid_artifact_rejected_and_failure_keeps_prior_model(self):
        fixture(self.data)
        train_neural(self.data, self.model, self.report)
        before = self.model.read_bytes()
        # A training failure before artifact replace must leave prior artifact intact.
        fixture(self.root / 'bad.csv', sessions=('only1', 'only2'))
        with self.assertRaises(ValueError): train_neural(self.root / 'bad.csv', self.model, self.report)
        self.assertEqual(before, self.model.read_bytes())
        bad = self.root / 'bad.npz'
        np.savez(bad, mean=np.zeros(63))
        with self.assertRaises(ValueError): load_neural(bad)

    def test_invalid_metadata_shapes_are_rejected(self):
        from neural_learning import LandmarkMLP
        model = LandmarkMLP(np.zeros(63), np.ones(63), [np.zeros(s) for s in ((63,128),(128,64),(64,32),(32,3))],
                            [np.zeros(n) for n in (128,64,32,3)])
        from neural_learning import _atomic_npz
        bad = self.root / 'null_meta.npz'
        # Build a valid-shaped artifact whose metadata has the wrong JSON type.
        arrays = {'mean': model.mean, 'scale': model.scale, 'metadata': np.asarray('null')}
        for i, (w,b) in enumerate(zip(model.weights, model.biases)):
            arrays[f'w{i}'] = w; arrays[f'b{i}'] = b
        np.savez(bad, **arrays)
        with self.assertRaises(ValueError): load_neural(bad)

    def test_sidecar_failure_keeps_authoritative_model_and_report(self):
        fixture(self.data)
        blocker = self.root / 'not_a_directory'
        blocker.write_text('occupied')
        model, report = train_neural(self.data, self.model, blocker / 'report.json')
        self.assertIn('report_sidecar_error', report)
        loaded = load_neural(self.model)
        self.assertEqual(loaded.metadata['evaluation']['accuracy'], report['accuracy'])
        np.testing.assert_allclose(loaded.predict_proba(np.zeros((2, 63))), model.predict_proba(np.zeros((2, 63))))


if __name__ == '__main__': unittest.main()
