# VisionPlay — Rock, Paper, Scissors with AI

A small webcam project combining **computer vision, pretrained deep learning, and a trainable classical machine-learning classifier**. Play immediately with Google's pretrained gesture recognizer; collect your own examples to demonstrate SVM training.

> **Python 3.13 installed?** Keep it. Install **64-bit Python 3.11 alongside it** using the link below. This project supports Python 3.10–3.12 only; setup automatically prefers 3.11.

## Start here — Windows

1. On GitHub, choose **Code → Download ZIP**, then **extract the whole ZIP** to a normal writable folder, such as `Documents\VisionPlay`. Do not run it from inside the ZIP.
2. Use **64-bit Python 3.10, 3.11, or 3.12**. If none is installed, the [Python 3.11.9 release page](https://www.python.org/downloads/release/python-3119/) has a compatible **Windows installer (64-bit)**. Enable the Python launcher in the installer. This package targets Windows x64; ARM and 32-bit Python are not supported.
3. Double-click **`setup.bat`**. Keep its window open until it reports success. Internet is needed to install Python packages and download the official Google model. Setup uses a project-local virtual environment; it does not need administrator access or change PowerShell policy.
4. Double-click **`runme.bat`**. Allow camera access if Windows asks. Click the video window so it receives keyboard input.
5. Show one hand, palm toward the camera, with good lighting. Click **Play** or press **Space**, and hold a pose through the three-second countdown.

After successful setup, the webcam demo runs locally without an Internet connection. No account or API key is needed. Do setup and a test round **before** your classroom demonstration.

### Your three moves

| Game move | Hand pose | Pretrained label |
|---|---|---|
| Rock | Closed fist | `Closed_Fist` |
| Paper | Open palm, fingers spread | `Open_Palm` |
| Scissors | Index and middle fingers in a V | `Victory` |

## Play a match

**First to three wins** takes the match: best of five **decisive** rounds. Ties do not count toward that limit, so a match can contain more than five rounds in total.

Click **Play** or press **Space**. The computer locks its random move **before** the three-second countdown and keeps it hidden until the reveal. Hold your gesture steady. If recognition is not stable at the countdown's end, you get up to **1.5 extra seconds** to settle your hand; if it is still not valid and stable, the round is canceled with **no score**. Start another round when ready. Click **New match** or press **R** to reset.

The computer does not predict your strategy. The screen shows both available classifier predictions; **M** chooses which one controls your move.

## Mouse and keyboard controls

Use the on-screen buttons, or click the window and use these shortcuts:

| Key | Action |
|---|---|
| **Space** | Play a round; in guided training, capture the current stage |
| **G** | Open guided sample collection |
| **C** | Capture the current stage in guided training |
| **T** | Train/evaluate the SVM from saved samples |
| **E** | Open teacher view: pipeline, held-out results, and confusion matrix |
| **M** | Switch pretrained/SVM game mode |
| **R** | Start a new match |
| **Escape** | Back from teacher/training view; stops an active capture without deleting saved samples |
| **Q** | Quit |

Guided collection uses explicit **Next** buttons between stages and sessions—there is no automatic advance into the next pose. The old **1/2/3/N** capture shortcuts are no longer used.

## Add your classical ML model — guided training

A trained SVM is not supplied: **you create it from real examples**. Pretrained mode works immediately after setup.

1. Click **Guided training** or press **G**. Follow the displayed pose: **rock → paper → scissors**.
2. Hold that pose and click **Capture** (or press **C/Space**). Each stage targets **40 detected-hand samples**. Wait for the preparation countdown, then keep the same pose throughout capture; gently vary position without changing the gesture.
3. Click **Next** when ready for the next pose. After all three poses, explicitly advance to the next session. Repeat for **three sessions**. Reposition your hand and vary distance or lighting moderately between sessions; use the same hand and palm orientation initially.
4. Use **Cancel capture** to stop a batch, or **Back/Escape** to leave training. Samples already written to CSV are preserved; canceling is not deletion. Use the screen's status to see the next step.
5. Click **Train** or press **T**. For compatibility with existing data, training requires at least **three complete sessions**, each with **20 or more examples of every class**; the new guided flow targets 40 per class per session.
6. Click **Teacher view** or press **E** for the pipeline and your measured results (also saved in `data/evaluation.json`). Return with **Back/Escape**, then switch to **SVM** using its button or **M** and play.

The newest complete session is held out; earlier complete sessions train the evaluation model. Incomplete sessions are excluded. After evaluation, the live SVM is refit on all complete sessions. The held-out score therefore describes the evaluation model, **not an independent test of the refitted live model**.

Nearby video frames are correlated. A session split reduces leakage but a small personal dataset does not establish accuracy for new people or conditions. Do not repeatedly tune against the same holdout; test improvements on a genuinely new session.

## What each subject contributes

- **Computer vision (OpenCV):** camera frames, image conversion, live display, and hand-landmark overlays.
- **Deep learning (MediaPipe):** pretrained neural models locate a hand, estimate 21 landmarks, and recognize supported gestures. No neural network is trained from scratch in this project.
- **Classical machine learning (scikit-learn):** an SVM learns your three gesture classes from normalized landmark coordinates. These are **deep-learning-extracted features**, not raw pixels. This is a hybrid pipeline, not a separately trained image CNN.
- **Ordinary programming:** countdown, random computer move, game rules, and scorekeeping. These are not machine learning.

See **`TEACHER_GUIDE.md`** for a one-minute demonstration script and viva questions.

**Verification:** the earlier webcam version was tested on Windows. This upgraded interface has been developed and checked in the cloud; that is not a Windows webcam acceptance test. Test this version on your laptop before class.

## Privacy and files

Webcam frames stay in memory on your laptop; this app does not record or upload images. Capturing examples writes labeled numerical hand landmarks locally in `data/samples.csv`. Training saves `data/evaluation.json`; the SVM is refit in memory from `data/samples.csv` on startup when sufficient data exists. Incomplete sessions are excluded from fitting and evaluation. No serialized SVM model is loaded. Network access is used by setup to obtain dependencies and Google's pretrained model, not to analyze your webcam.

## How the virtual environment works

A venv isolates packages; it **cannot install Python 3.11 itself**. Install the base interpreter once, side by side with any newer Python. `setup.bat` checks the Python launcher, standard per-user Python folders, and executables on PATH. It then creates `.venv` in this project and installs packages only there. You do not need to activate it: both BAT files use its interpreter directly. A valid existing environment is reused.

If an old `.venv` is broken or uses an unsupported version, setup stops without deleting anything. Close the app, rename `.venv` to `.venv-backup` in File Explorer, and rerun setup. This preserves `data/` samples and the downloaded `models/` file. Backup environments are ignored by Git. Do not copy a venv between computers or move it out of the project; recreate it instead.

## Troubleshooting

- **No supported Python:** install x64 Python 3.10, 3.11, or 3.12, then rerun `setup.bat`. Python 3.13+ is outside this package's pinned support range. `py -0p` in Command Prompt lists launcher installations.
- **Setup fails:** read the final error. Check Internet access and available disk space; rerun setup. Do not disable antivirus or run as administrator as a workaround. If a school network blocks package/model downloads, complete setup on an allowed network before class.
- **Camera unavailable:** close Teams, Zoom, browsers, and other camera apps. In Windows Settings → Privacy & security → Camera, allow desktop apps to access the camera. To try camera index 1, open Command Prompt in this folder and run `runme.bat 1` (default is 0).
- **Window opens then closes:** run `runme.bat` from Command Prompt and read the error rather than repeatedly double-clicking.
- **Keys do nothing:** click the app window to give it focus, or use its on-screen buttons. Capture shortcuts only apply in guided training.
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
