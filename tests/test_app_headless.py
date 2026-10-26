"""Integration with stub camera/model only; not evidence of webcam accuracy."""
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import app

class AppFlowTests(unittest.TestCase):
    def run_flow(self, keys, closed=False, camera_ok=True):
        cap=MagicMock();cap.isOpened.return_value=True
        cap.read.return_value=(camera_ok,np.zeros((480,640,3),dtype=np.uint8))
        detector=MagicMock();detector.recognize_for_video.return_value=SimpleNamespace(hand_landmarks=[])
        with tempfile.TemporaryDirectory() as temp, patch.object(app,'DATA',Path(temp)/'samples.csv'), patch.object(app,'REPORT',Path(temp)/'evaluation.json'), patch.object(app,'recognizer',return_value=detector), patch('cv2.VideoCapture',return_value=cap), patch('cv2.namedWindow'), patch('cv2.resizeWindow'), patch('cv2.setMouseCallback'), patch('cv2.imshow'), patch('cv2.destroyAllWindows'), patch('cv2.waitKey',side_effect=keys), patch('cv2.getWindowProperty',return_value=0 if closed else 1):
            if not camera_ok:
                with self.assertRaisesRegex(RuntimeError,'disconnected'):app.main(0)
            else:app.main(0)
            self.assertFalse((Path(temp)/'samples.csv').exists())
        cap.release.assert_called();detector.close.assert_called_once()
        return cap
    def test_teacher_back_training_cancel_quit(self):
        self.run_flow([ord('e'),27,ord('g'),ord('c'),27,ord('q')])
    def test_window_close_wins_over_space(self):
        cap=self.run_flow([ord(' ')],closed=True)
        self.assertEqual(cap.read.call_count,1)
    def test_camera_loss_releases_resources(self):
        self.run_flow([],camera_ok=False)

if __name__=='__main__':unittest.main()
