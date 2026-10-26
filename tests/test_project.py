"""Synthetic software tests only; these are NOT gesture accuracy measurements."""
import unittest
import runpy
from unittest.mock import patch
import tempfile
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import features, outcome, LABELS
from learning import add_sample, train

class ProjectTests(unittest.TestCase):
    def test_python_version_guard(self):
        script = Path(__file__).resolve().parents[1] / 'check_python.py'
        with patch.object(sys, 'version_info', (3, 13, 0)):
            with self.assertRaises(SystemExit) as exc: runpy.run_path(str(script))
            self.assertEqual(exc.exception.code, 1)
    def test_venv_guard(self):
        script = Path(__file__).resolve().parents[1] / 'check_python.py'
        with patch.object(sys, 'argv', [str(script), '--venv']), patch.object(sys, 'version_info', (3, 11, 9)), patch.object(sys, 'prefix', sys.base_prefix):
            with self.assertRaises(SystemExit) as exc: runpy.run_path(str(script))
            self.assertEqual(exc.exception.code, 1)
    def test_outcomes(self):
        expected = [['draw','lose','win'],['win','draw','lose'],['lose','win','draw']]
        for i,a in enumerate(LABELS):
            for j,b in enumerate(LABELS): self.assertEqual(outcome(a,b),expected[i][j])
        with self.assertRaises(ValueError): outcome('bad','rock')
    def test_features(self):
        p = np.arange(63,dtype=float).reshape(21,3)
        np.testing.assert_allclose(features(p),features(p*3+7))
        self.assertEqual(features(p).shape,(63,))
        with self.assertRaises(ValueError): features(np.zeros((21,3)))
        with self.assertRaises(ValueError): features(np.zeros((20,3)))
    def test_session_holdout(self):
        with tempfile.TemporaryDirectory() as temp:
            csv = Path(temp)/'samples.csv'
            report_path = Path(temp)/'evaluation.json'
            rng = np.random.default_rng(5)
            for s in ['a','b','c']:
                for i,label in enumerate(LABELS):
                    for _ in range(20): add_sample(csv,s,label,rng.normal(i,0.05,63))
            add_sample(csv,'incomplete','rock',np.zeros(63))
            model, report = train(csv,report_path)
            self.assertEqual(report['train_sessions'],['a','b'])
            self.assertEqual(report['test_session'],'c')
            self.assertEqual(report['train_samples'],120)
            self.assertEqual(report['test_samples'],60)
            self.assertEqual(report['excluded_incomplete_samples'],1)
            self.assertTrue(report_path.exists())
            self.assertEqual(len(model.predict(np.zeros((1,63)))),1)
    def test_insufficient_data(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError): train(Path(temp)/'missing.csv',Path(temp)/'r.json')

if __name__ == '__main__': unittest.main()
