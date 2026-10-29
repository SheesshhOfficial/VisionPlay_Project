"""Controller/UI neural-mode regressions; no camera and no accuracy claims."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from concurrent.futures import Future
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from controller import Controller, Stability
from ui import render
from app import finish_training_result, load_neural_artifact, safe_feature_vector


class NeuralControllerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.c = Controller(Path(self.tmp.name) / 'samples.csv')

    def test_unavailable_neural_mode_does_not_switch(self):
        self.c.action('mode', 0, svm_ready=False, neural_ready=False)
        self.assertEqual(self.c.mode, 'PRETRAINED DL')
        self.assertIn('Train', self.c.note)

    def test_neural_mode_available_and_locked_during_round(self):
        self.c.action('mode', 0, neural_ready=True)
        self.assertEqual(self.c.mode, 'YOUR NEURAL NET')
        self.c.action('play', 1)
        self.c.action('mode', 1.1, svm_ready=True, neural_ready=True)
        self.assertEqual(self.c.mode, 'YOUR NEURAL NET')
        self.assertIn('Finish', self.c.note)

    def test_start_and_mode_switch_blocked_during_training(self):
        self.c.training_job = 'neural'
        self.c.action('play', 0)
        self.assertFalse(self.c.game.active)
        self.assertIn('Wait', self.c.note)
        self.c.action('mode', 1, neural_ready=True)
        self.assertEqual(self.c.mode, 'PRETRAINED DL')

    def test_async_job_guard_and_failure_keeps_prior_reference(self):
        self.c.action('training', 0)
        self.c.wizard.completed_sessions = 3
        self.assertEqual(self.c.action('train_neural', 1), 'train_neural')
        self.assertIsNone(self.c.action('train', 1.1))
        self.assertIsNone(self.c.action('capture', 1.2))
        # Result/error harvesting is owned by app's UI thread; controller job state
        # remains a simple guard until that thread calls training_finished().
        self.assertEqual(self.c.training_job, 'neural')
        self.c.training_finished()
        self.assertIsNone(self.c.training_job)

    def test_background_success_applies_model_and_error_preserves_previous(self):
        previous = object()
        self.c.training_job = 'neural'
        done = Future(); trained = object()
        report = {'accuracy': .75, 'training_loss_curve': [.8, .2]}
        done.set_result((trained, report))
        model, svm_report, neural, evaluation, status = finish_training_result(
            self.c, done, 'neural', None, None, previous, None, 'old status')
        self.assertIs(neural, trained)
        self.assertEqual(evaluation, report)
        self.assertEqual(status, 'ready')
        self.c.training_job = 'neural'
        failed = Future(); failed.set_exception(ValueError('bad dataset'))
        _, _, kept, _, _ = finish_training_result(
            self.c, failed, 'neural', None, None, trained, report, 'ready')
        self.assertIs(kept, trained)
        self.assertIn('previous model kept', self.c.note)

    def test_navigation_during_training_does_not_switch_game_mode(self):
        self.c.action('training', 0)
        self.c.wizard.completed_sessions = 3
        self.c.action('train_neural', 1)
        self.c.action('back', 1.1)
        self.c.training_finished()  # completed result is harvested only by UI thread
        self.assertEqual(self.c.screen, 'play')
        self.assertEqual(self.c.mode, 'PRETRAINED DL')

    def test_stale_prediction_is_cleared_when_hand_disappears(self):
        gate = Stability()
        for t in (0, .15, .4): value, _ = gate.update(t, 'rock')
        self.assertEqual(value, 'rock')
        value, progress = gate.update(.41, None)
        self.assertIsNone(value)
        self.assertEqual(progress, 0)
        gate.clear()
        self.assertIsNone(gate.update(2, 'rock')[0])

    def test_ui_exposes_neural_status_and_neural_train_action(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        canvas, buttons = render(frame, {'screen': 'training', 'training': {'complete_sessions': 3}, 'neural_status': 'ready'})
        self.assertIn('train_neural', buttons)
        self.assertEqual(canvas.shape, (800, 1280, 3))

    def test_bad_landmarks_are_safely_rejected(self):
        points=[SimpleNamespace(x=float('nan'),y=0.,z=0.) for _ in range(21)]
        self.assertIsNone(safe_feature_vector(points))
        self.assertIsNone(safe_feature_vector([SimpleNamespace(x=0,y=0)]))

    def test_embedded_report_wins_over_stale_sidecar(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); model_path=root/'neural_model.npz'; model_path.touch()
            data_path=root/'samples.csv'; data_path.write_text('new data')
            embedded={'accuracy':.8,'test_samples':99}
            model=SimpleNamespace(metadata={'evaluation':embedded,'dataset_sha256':'old'})
            with patch('neural_learning.load_neural', return_value=model):
                loaded,report,note=load_neural_artifact(model_path,data_path)
            self.assertIs(loaded,model)
            self.assertIs(report,embedded)
            self.assertIn('older than samples',note)


if __name__ == '__main__':
    unittest.main()
