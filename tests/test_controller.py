import sys
from pathlib import Path
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from controller import Controller, Stability

class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.app=Controller(Path(self.temp.name)/'samples.csv')
    def test_repeat_play_click_preserves_locked_round(self):
        self.app.action('play',0)
        first=self.app.game.countdown_remaining(1)
        self.app.action('play',1)
        self.assertEqual(self.app.game.countdown_remaining(1),first)
        self.assertIsNone(self.app.game.computer_move)
    def test_navigation_cancels_round_without_score(self):
        self.app.action('play',0)
        self.app.action('teacher',1)
        self.assertFalse(self.app.game.active)
        self.assertEqual(self.app.game.player_wins,0)
        self.app.action('back',2)
        self.assertEqual(self.app.screen,'play')
    def test_training_back_preserves_data_and_pauses(self):
        self.app.action('training',0)
        self.app.action('capture',0)
        self.app.wizard.tick(2,np.ones(63))
        self.app.action('back',2.1)
        self.assertFalse(self.app.wizard.active)
        self.assertEqual(self.app.wizard.count,1)
        self.app.action('training',3)
        self.app.action('capture',3)
        self.assertTrue(self.app.wizard.active)
    def test_camera_loss_cancels(self):
        self.app.action('play',0);self.app.camera_lost()
        self.assertFalse(self.app.running)
        self.assertFalse(self.app.game.active)
        self.assertEqual(self.app.game.player_wins,0)
    def test_quit_pauses_capture(self):
        self.app.action('training',0);self.app.action('capture',0)
        self.app.action('quit',1)
        self.assertFalse(self.app.wizard.active)
        self.assertFalse(self.app.running)
    def test_mode_switch_during_round_ignored(self):
        self.app.action('play',0);self.app.action('mode',1,True)
        self.assertEqual(self.app.mode,'PRETRAINED DL')
    def test_two_frames_do_not_fill_stability_bar(self):
        gate=Stability();gate.update(0,"rock")
        label,progress=gate.update(0.4,"rock")
        self.assertIsNone(label);self.assertLess(progress,1)
    def test_stability_never_reuses_missing_hand(self):
        stable=Stability()
        for t in (0,0.1,0.2,0.4): label,progress=stable.update(t,'rock')
        self.assertEqual(label,'rock')
        label,progress=stable.update(0.41,None)
        self.assertIsNone(label)
        self.assertEqual(progress,0)
        stable.clear()
        self.assertIsNone(stable.update(1,'rock')[0])

if __name__=='__main__':unittest.main()
