"""Synthetic capture-controller tests, not model accuracy measurements."""
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_state import TrainingWizard
from learning import add_sample, read_samples
from core import LABELS


class WizardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'samples.csv'
        self.wizard = TrainingWizard(self.path)
        self.vector = np.arange(63, dtype=float)

    def tearDown(self):
        self.temp.cleanup()

    def batch(self, now):
        self.assertTrue(self.wizard.begin(now))
        for i in range(40):
            self.wizard.tick(now + 2 + i * .21, self.vector)
        return now + 12

    def test_repeated_start_does_not_reset_preparation(self):
        self.assertTrue(self.wizard.begin(0))
        self.assertFalse(self.wizard.begin(1))
        self.assertEqual(self.wizard.ready_at, 2)
        self.wizard.tick(1, self.vector)
        self.assertEqual(self.wizard.count, 0)
        self.wizard.tick(2, self.vector)
        self.wizard.tick(2.01, self.vector)
        self.assertEqual(self.wizard.count, 1)

    def test_missing_hand_timeout_and_no_invented_samples(self):
        self.wizard.begin(0)
        self.wizard.tick(3, None)
        self.assertFalse(self.path.exists())
        self.wizard.tick(33, self.vector)
        self.assertEqual(self.wizard.phase, 'ready')
        self.assertEqual(self.wizard.count, 0)

    def test_cancel_back_close_preserves_and_resumes(self):
        self.wizard.begin(0)
        self.wizard.tick(2, self.vector)
        session = self.wizard.session_id
        self.assertTrue(self.wizard.cancel())
        self.assertFalse(self.wizard.cancel())
        self.wizard.tick(10, self.vector)
        self.assertEqual(self.wizard.count, 1)
        self.wizard.begin(20)
        self.wizard.tick(22, self.vector)
        self.assertEqual(self.wizard.count, 2)
        self.assertEqual(self.wizard.session_id, session)
        self.assertEqual(len(read_samples(self.path)[0]), 2)

    def test_explicit_pose_and_session_transitions(self):
        now = self.batch(0)
        self.assertEqual(self.wizard.phase, 'batch_done')
        self.assertEqual(self.wizard.label, 'rock')
        self.wizard.tick(now, self.vector)
        self.assertEqual(self.wizard.count, 40)
        now = self.batch(now)
        now = self.batch(now)
        self.assertEqual(self.wizard.phase, 'session_done')
        self.assertEqual(self.wizard.completed_sessions, 1)
        old = self.wizard.session_id
        self.wizard.begin(now)
        self.assertNotEqual(self.wizard.session_id, old)
        self.assertEqual(self.wizard.label, 'rock')
        self.assertEqual(self.wizard.session_number, 2)

    def test_three_sessions_ready_and_existing_data_preserved(self):
        now = 0
        for _ in range(9):
            now = self.batch(now)
        self.assertEqual(self.wizard.phase, 'complete')
        self.assertTrue(self.wizard.training_ready)
        self.assertEqual(len(read_samples(self.path)[0]), 360)
        restored = TrainingWizard(self.path)
        self.assertEqual(restored.completed_sessions, 3)
        self.assertTrue(restored.training_ready)
        self.assertNotEqual(restored.session_id, self.wizard.session_id)
        self.assertEqual(len(read_samples(self.path)[0]), 360)

    def test_legacy_complete_sessions_and_invalid_vectors(self):
        for label in LABELS:
            for _ in range(20):
                add_sample(self.path, 'legacy', label, self.vector)
        restored = TrainingWizard(self.path)
        self.assertEqual(restored.completed_sessions, 1)
        self.wizard.begin(0)
        with self.assertRaises(ValueError): self.wizard.tick(2, [1, 2])
        self.assertEqual(self.wizard.count, 0)

if __name__ == '__main__':
    unittest.main()
