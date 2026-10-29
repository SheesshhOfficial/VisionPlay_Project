# VisionPlay — Teacher guide

**Aim:** recognize rock, paper, and scissors from a webcam, combining pretrained deep learning with two trainable models: a classical SVM and an experimental neural MLP.

## How it works

```text
Webcam → OpenCV frames → MediaPipe pretrained neural models
                                 ↓
                  21 hand landmarks + gesture label
                        /                 \
           PRETRAINED MODE                 USER-TRAINED MODES
         Google's gesture label       normalized landmark coordinates
                                        ↙                 ↘
                               SVM classifier       Landmark MLP
                                        \             /
                      game rules → scoreboard
```

| Area | Role |
|---|---|
| Computer vision | OpenCV reads frames, converts images, and draws the hand skeleton/UI. |
| Pretrained deep learning | Google's pretrained neural models detect the hand, estimate landmarks, and classify supported gestures. |
| Classical ML | An SVM learns rock/paper/scissors from your labeled normalized landmarks. |
| Trained neural model | A small scikit-learn MLP learns the same classes from 63 normalized landmark coordinates (21 xyz points). It is not a raw-pixel CNN. |
| Ordinary code | Countdown, random computer move, game rules, and scorekeeping. |

**Important:** both user-trained classifiers use **DL-extracted landmark features**, not raw webcam pixels. The MLP is trained from scratch on these 63 values; MediaPipe's pretrained neural models remain responsible for hand detection and landmarks.

## Before class

Run `setup.bat`, then `runme.bat`, and test **this upgraded version** with your webcam before class. The previous version was webcam-tested on Windows; cloud checks of this upgrade do not replace a Windows webcam test.

Click **Guided training** or press **G**. Capture 40 samples for **rock → paper → scissors**, using the **Capture** button or **C/Space**. Click **Next** between stages and sessions. Complete **three sessions**. Use **T** to train the SVM or **Train neural / N** for the MLP. Both paths require at least 20 examples of every class in each of three complete sessions.

Hold the assigned pose for a whole batch. Reposition yourself between sessions and vary distance or lighting moderately. **Cancel capture** or **Back/Escape** stops collection but preserves samples already saved. Old numbered **1/2/3** capture shortcuts are removed. **N** is now reserved for neural training; use the on-screen Next controls instead of the former N-to-advance shortcut.

Training runs in a background worker so the camera window remains responsive. While it runs, the app blocks starting a new round, capturing more samples, and switching game mode. Closing the window releases the camera; a fit already underway may finish saving its artifact before Python exits.

## One-minute demonstration

**0–15 seconds:** “This is VisionPlay. OpenCV handles my webcam, and Google's pretrained deep-learning models detect my hand and its 21 landmarks.” Show a fist, open palm, and V sign.

**15–30 seconds:** “In pretrained mode, I use Google's gesture labels.” Click **Play** or press **Space**. “The computer locks a hidden random move before the countdown, then reveals it. First to three wins; ties do not count.”

**30–45 seconds:** Use **M** to cycle through classifiers that are available in this build. “The SVM is the classical ML model; the experimental MLP is a small neural model trained on 63 normalized landmark values, not photographs.” Say this only if you collected data and completed training. The MLP uses a 0.65 confidence cutoff for a move; that is a decision threshold, **not** a calibrated accuracy estimate.

**45–60 seconds:** Click **Teacher view** or press **E** to show validated results. SVM metrics are in `data/evaluation.json`; neural metrics and its loss curve, if actually produced, are in `data/neural_evaluation.json`. “The newest complete session was held out. This personal result is not a guarantee for other people or conditions.”

In Teacher View, **V** switches between SVM and neural evaluation when a neural report exists. Return with **Back/Escape**. If a pose is unstable after the three-second countdown, the game waits up to 1.5 seconds longer; if still invalid, it cancels with no score. **R/New match** resets the match.

Never claim a model was trained or webcam-tested unless that actually happened. A safe description before a real training run is: **“The training workflow is implemented; this build has not yet produced a real-data result.”**

## Explain your results honestly

- Earlier complete sessions train the evaluation model; the newest complete session is held out. Incomplete sessions are excluded.
- The MLP is `StandardScaler + MLPClassifier(hidden_layer_sizes=(128,64,32), activation='relu', solver='adam')`, fitted on the earlier complete sessions only. Its `data/neural_model.npz` stores the exact held-out-tested parameters; unlike SVM behavior, it is not refit on all sessions.
- The model archive embeds the authoritative evaluation report, including metrics and loss curve; `data/neural_evaluation.json` is a convenience sidecar that may not be present if its write fails. Both reflect a real successful run only. Do not present confidence as accuracy.
- At game startup the app loads `data/neural_model.npz` if present; it does not retrain. Training through the wizard writes the tested artifact, and the game can use the loaded/trained model in its available-mode cycle.
- Report accuracy with the test-sample count. The confusion matrix has **true classes in rows and predicted classes in columns**, in the `rock`, `paper`, `scissors` order.
- For the SVM, a separate live model is refit on all complete sessions after evaluation. Its accuracy is **not independently measured** by that holdout score. The exported MLP is not refit.
- Do not repeatedly tune against the same holdout. Test improvements on a genuinely new session.
- A small one-person dataset does not establish performance on new people, hands, cameras, or lighting. Poor results are still valid findings—explain the likely cause and improvement.
- No accuracy is claimed before you collect real samples. Synthetic software tests are not webcam-performance measurements.

## Quick viva questions

**What was trained?**
MediaPipe supplies pretrained neural models. Separately, the project can train a small sklearn MLP on landmark vectors and a classical SVM on those same features. The MLP is not a raw-image CNN. State which workflow you actually ran; confidence is not accuracy.

**Why landmarks?**  
They give a compact hand-shape representation, making quick CPU training with a small dataset practical. The tradeoff is dependence on the pretrained landmark detector.

**Why normalize?**  
To reduce sensitivity to hand position and overall size. It does not solve every rotation, viewpoint, or handedness difference.

**Is deep learning also machine learning?**  
Yes. Here “pretrained DL” refers to Google's supplied hand/gesture model; “neural MLP” refers to the small model fitted on landmark features; “classical ML” refers to the separate SVM.

**Why split by session?**  
Neighboring frames are similar. Randomly mixing them into training and testing can inflate the score. A session split reduces that leakage, but does not prove cross-person performance.

**Can it recognize anything?**  
No. The game has three classes. The SVM may assign a known class to an unfamiliar gesture. Unknown-gesture rejection and multi-person testing are good next steps.

**Are images stored or uploaded?**  
No. Frames stay local in memory; optional capture saves only labeled landmark numbers to CSV.

## Attribution

Google MediaPipe supplies pretrained models, OpenCV the vision interface, and scikit-learn the SVM and MLP implementations. The MLP architecture and training procedure are this project's own use of sklearn's standard estimator. For fair play, the computer's move is chosen randomly before the countdown; the CPU opponent does not use learned strategy. Concept inspiration: [smoorjani/rock-paper-scissor](https://github.com/smoorjani/rock-paper-scissor). This is a new implementation; no code or reported accuracy from that repository is presented as this project's work. Official documentation links are in README.
