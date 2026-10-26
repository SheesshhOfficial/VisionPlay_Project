# VisionPlay — Teacher guide

**Aim:** recognize rock, paper, and scissors from a webcam, combining pretrained deep learning with a small classical ML model you train yourself.

## How it works

```text
Webcam → OpenCV frames → MediaPipe pretrained neural models
                                 ↓
                  21 hand landmarks + gesture label
                        /                 \
           PRETRAINED MODE              SVM MODE
         Google's gesture label    normalized landmark coordinates
                        \           → user-trained SVM
                         \               /
                      game rules → scoreboard
```

| Area | Role |
|---|---|
| Computer vision | OpenCV reads frames, converts images, and draws the hand skeleton/UI. |
| Deep learning | Google's pretrained neural models detect the hand, estimate landmarks, and classify supported gestures. |
| Classical ML | An SVM learns rock/paper/scissors from your labeled, normalized landmark coordinates. |
| Ordinary code | Countdown, random computer move, game rules, and scorekeeping. |

**Important:** the SVM uses **DL-extracted features**. It is not an independent raw-image classifier or a CNN trained from scratch.

## Before class

Run `setup.bat`, then `runme.bat`, and test the webcam. For an explicit classical-ML demonstration, collect all three gestures in **three sessions**: keys **1/2/3** capture rock/paper/scissors, **N** starts another session, and **T** trains. Each session needs at least 20 examples of every class. A batch targets 40 samples after a two-second preparation delay.

Hold the correct gesture for the whole batch. Between sessions, reposition yourself and vary distance or lighting moderately. Do not treat nearly identical consecutive frames as independent evidence.

## One-minute demonstration

**0–15 seconds:** “This is VisionPlay. OpenCV handles my webcam, and Google's pretrained deep-learning models detect my hand and its 21 landmarks.” Show a fist, open palm, and V sign.

**15–30 seconds:** “In pretrained mode, I use Google's gesture labels.” Press **Space** and play a round. “The computer chooses randomly; it does not predict my strategy.”

**30–45 seconds:** Press **M**. “This mode uses an SVM I trained on my own labeled landmark samples. It learns from numerical hand-shape features, not raw photographs.” Only say this if you actually collected data and trained it.

**45–60 seconds:** Press **E** to show the evaluation panel (also saved as `data/evaluation.json`). “I held out a complete capture session for evaluation rather than mixing adjacent frames. These are my measured results, not a guarantee for other people or conditions.”

If no SVM has been trained, say: **“The SVM workflow is implemented, but I have not trained it yet.”** The pretrained game alone does not demonstrate your own classical classifier.

## Explain your results honestly

- Earlier complete sessions train the evaluation model; the newest complete session is held out. Incomplete sessions are excluded.
- Report accuracy with the test-sample count. The confusion matrix has **true classes in rows and predicted classes in columns**, in the report's `classes` order.
- After evaluation, a separate live model is refit on all complete sessions. Its accuracy is **not independently measured** by that holdout score.
- Do not repeatedly tune against the same holdout. Test improvements on a genuinely new session.
- A small one-person dataset does not establish performance on new people, hands, cameras, or lighting. Poor results are still valid findings—explain the likely cause and improvement.
- No accuracy is claimed before you collect real samples. Synthetic software tests are not webcam-performance measurements.

## Quick viva questions

**Did you train the neural network?**  
No. Google supplies the pretrained neural models. I train the classical SVM.

**Why landmarks?**  
They give a compact hand-shape representation, making quick CPU training with a small dataset practical. The tradeoff is dependence on the pretrained landmark detector.

**Why normalize?**  
To reduce sensitivity to hand position and overall size. It does not solve every rotation, viewpoint, or handedness difference.

**Is deep learning also machine learning?**  
Yes. Here “classical ML” means the SVM, while “DL” means the neural models.

**Why split by session?**  
Neighboring frames are similar. Randomly mixing them into training and testing can inflate the score. A session split reduces that leakage, but does not prove cross-person performance.

**Can it recognize anything?**  
No. The game has three classes. The SVM may assign a known class to an unfamiliar gesture. Unknown-gesture rejection and multi-person testing are good next steps.

**Are images stored or uploaded?**  
No. Frames stay local in memory; optional capture saves only labeled landmark numbers to CSV.

## Attribution

Google MediaPipe supplies pretrained models, OpenCV the vision interface, and scikit-learn the SVM. Concept inspiration: [smoorjani/rock-paper-scissor](https://github.com/smoorjani/rock-paper-scissor). This is a new implementation; no code or reported accuracy from that repository is presented as this project's work. Official documentation links are in README.
