# VisionPlay — Rock, Paper, Scissors with AI

A small webcam project combining **computer vision, pretrained deep learning, and a trainable classical machine-learning classifier**. Play immediately with Google's pretrained gesture recognizer; collect your own examples to demonstrate SVM training.

> **Python 3.13 installed?** Keep it. Install **64-bit Python 3.11 alongside it** using the link below. This project supports Python 3.10–3.12 only; setup automatically prefers 3.11.

## Start here — Windows

1. On GitHub, choose **Code → Download ZIP**, then **extract the whole ZIP** to a normal writable folder, such as `Documents\VisionPlay`. Do not run it from inside the ZIP.
2. Use **64-bit Python 3.10, 3.11, or 3.12**. If none is installed, the [Python 3.11.9 release page](https://www.python.org/downloads/release/python-3119/) has a compatible **Windows installer (64-bit)**. Enable the Python launcher in the installer. This package targets Windows x64; ARM and 32-bit Python are not supported.
3. Double-click **`setup.bat`**. Keep its window open until it reports success. Internet is needed to install Python packages and download the official Google model. Setup uses a project-local virtual environment; it does not need administrator access or change PowerShell policy.
4. Double-click **`runme.bat`**. Allow camera access if Windows asks. Click the video window so it receives keyboard input.
5. Show one hand, palm toward the camera, with good lighting. Press **Space** and hold a pose through the three-second countdown.

After successful setup, the webcam demo runs locally without an Internet connection. No account or API key is needed. Do setup and a test round **before** your classroom demonstration.

### Your three moves

| Game move | Hand pose | Pretrained label |
|---|---|---|
| Rock | Closed fist | `Closed_Fist` |
| Paper | Open palm, fingers spread | `Open_Palm` |
| Scissors | Index and middle fingers in a V | `Victory` |

The computer chooses a random move. This is not an AI that predicts your strategy.

## Controls

| Key | Action |
|---|---|
| **Space** | Start a three-second round |
| **M** | Switch between pretrained recognition and your trained SVM |
| **1 / 2 / 3** | Capture a batch for rock / paper / scissors |
| **N** | Start a new capture session |
| **T** | Train and evaluate the SVM from your recorded examples |
| **E** | Show/hide held-out evaluation score, counts, and confusion matrix |
| **R** | Reset game scores |
| **Q** | Quit |

The screen shows both available predictions; **M** chooses which classifier controls the game.

A trained SVM is not supplied: **you create it from real examples**. Pretrained mode works first, without collecting data.

## Add the classical ML demonstration — about 5 minutes

Training only needs landmark coordinates, not stored photographs.

1. Start a capture session. Hold **rock**, press **1**, and keep the pose while moving your hand gently within view. A batch targets **40 samples**, after a two-second preparation delay, approximately eight seconds when the hand remains detected. Do not change to another gesture during a batch.
2. Repeat with **paper → 2** and **scissors → 3**.
3. Press **N** for a new session. Change hand position/distance or lighting a little, and repeat all three batches. Use the same hand and palm orientation initially.
4. Press **N** again and collect all three gestures a third time. Make this a realistic independent test: lower your hand between batches and reposition it, rather than recording one continuous pose across sessions.
5. Press **T**. Training requires **at least three complete sessions**, each with **at least 20 examples of every class**. The newest complete session is held out for evaluation; earlier sessions train the evaluation model.
6. Press **E** to inspect the on-screen evaluation, or read `data/evaluation.json`. Press **M** to try the SVM in the game.

The evaluation model never trains on its held-out session. After evaluation, a separate deployment model is refit on all complete sessions so the game can use all complete sessions. Therefore the held-out score describes the evaluation model, **not an independent test of the refitted deployment model**. Collect a genuinely new session for further testing.

Do not present a high score as universal accuracy: nearby video frames are correlated, and your samples probably represent one person, one camera, and a few lighting conditions. A session split reduces leakage but does not establish performance for new people.

## What each subject contributes

- **Computer vision (OpenCV):** camera frames, image conversion, live display, and hand-landmark overlays.
- **Deep learning (MediaPipe):** pretrained neural models locate a hand, estimate 21 landmarks, and recognize supported gestures. No neural network is trained from scratch in this project.
- **Classical machine learning (scikit-learn):** an SVM learns your three gesture classes from normalized landmark coordinates. These are **deep-learning-extracted features**, not raw pixels. This is a hybrid pipeline, not a separately trained image CNN.
- **Ordinary programming:** countdown, random computer move, game rules, and scorekeeping. These are not machine learning.

See **`TEACHER_GUIDE.md`** for a short explanation and demonstration script.

## Privacy and files

Webcam frames stay in memory on your laptop; this app does not record or upload images. Capturing examples writes labeled numerical hand landmarks locally in `data/samples.csv`. Training saves `data/evaluation.json`; the SVM is refit in memory from `data/samples.csv` on startup when sufficient data exists. Incomplete sessions are excluded from fitting and evaluation. No serialized SVM model is loaded. Network access is used by setup to obtain dependencies and Google's pretrained model, not to analyze your webcam.

## Troubleshooting

- **No supported Python:** install x64 Python 3.10, 3.11, or 3.12, then rerun `setup.bat`. Python 3.13+ is outside this package's pinned support range. `py -0p` in Command Prompt lists launcher installations.
- **Setup fails:** read the final error. Check Internet access and available disk space; rerun setup. Do not disable antivirus or run as administrator as a workaround. If a school network blocks package/model downloads, complete setup on an allowed network before class.
- **Camera unavailable:** close Teams, Zoom, browsers, and other camera apps. In Windows Settings → Privacy & security → Camera, allow desktop apps to access the camera. To try camera index 1, open Command Prompt in this folder and run `runme.bat 1` (default is 0).
- **Window opens then closes:** run `runme.bat` from Command Prompt and read the error rather than repeatedly double-clicking.
- **Keys do nothing:** click the live video window. Use the number keys above the letters if the numeric keypad is unreliable.
- **No gesture / unstable recognition:** use one hand, show the full palm and fingertips, improve front lighting, move away from the frame edge, and hold the gesture still for the countdown. Other gestures are not valid game moves.
- **SVM unavailable:** first collect three complete sessions and press T. Incomplete or unbalanced recordings cannot produce an honest three-class evaluation.
- **SVM mistakes:** use the hand/orientation you trained with, add varied correctly labeled examples, then retrain. The SVM can misclassify gestures outside its three trained classes; its decision is not proof of certainty.
- **Slow video:** close heavy applications and avoid very dim lighting. Performance depends on your laptop; no FPS is guaranteed.

## Sources

- [Google MediaPipe Gesture Recognizer: Python API](https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer/python)
- [Google MediaPipe Gesture Recognizer model overview](https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer)
- [scikit-learn SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
- [OpenCV Python documentation](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)

This is an educational demo, not a production gesture-recognition system. Dependency and model licenses remain with their respective owners.

Concept inspiration: [smoorjani/rock-paper-scissor](https://github.com/smoorjani/rock-paper-scissor). This package is a new implementation, not a copy; no performance claims from that repository are reused.
