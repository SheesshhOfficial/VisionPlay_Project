"""VisionPlay: local webcam rock-paper-scissors with DL and trainable SVM."""
from pathlib import Path
from collections import deque, Counter
import argparse
import math
import secrets
import time
import uuid
import sys
import numpy as np
from core import LABELS, GESTURES, features, outcome
from learning import add_sample, train
ROOT = Path(__file__).resolve().parent
MODEL = ROOT / 'models' / 'gesture_recognizer.task'
DATA = ROOT / 'data' / 'samples.csv'
REPORT = ROOT / 'data' / 'evaluation.json'
EDGES = ((0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(0,17),(17,18),(18,19),(19,20))

def recognizer():
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    if not MODEL.exists(): raise RuntimeError('Missing model: run setup.bat first.')
    return vision.GestureRecognizer.create_from_options(vision.GestureRecognizerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=vision.RunningMode.VIDEO, num_hands=1,
        min_hand_detection_confidence=0.6, min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6))

def self_test():
    import cv2
    import mediapipe as mp
    with recognizer() as detector:
        result = detector.recognize_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.zeros((240,320,3),dtype=np.uint8)), 1)
        assert not result.hand_landmarks
    assert outcome('rock','scissors') == 'win'
    print('PASS: dependencies, official model load, blank-frame inference, game logic.')
    print('Camera and Windows GUI still need the live runme.bat test.')

def main(camera):
    import cv2
    import mediapipe as mp
    detector = recognizer()
    cap = cv2.VideoCapture(camera, cv2.CAP_DSHOW) if sys.platform == 'win32' else cv2.VideoCapture(camera)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        detector.close()
        raise RuntimeError('Cannot open webcam. Close Teams/Zoom, enable camera privacy access; try runme.bat 1.')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    window = 'VisionPlay | CV + Machine Learning + Deep Learning'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1100, 680)
    session = uuid.uuid4().hex[:12]
    model = None
    report = None
    show_eval = False
    note = 'Ready! SPACE to play. Keep one hand clearly visible.'
    if DATA.exists():
        try:
            model, report = train(DATA, REPORT)
            note = 'Your SVM was rebuilt from saved samples. M switches mode.'
        except ValueError: pass
    mode = 'PRETRAINED DL'
    capture = None
    round_end = None
    computer = None
    history = deque()
    scores = Counter()
    last_ms = 0
    last_tick = time.monotonic()
    fps = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok: raise RuntimeError('Webcam stopped returning frames. Reconnect it and restart.')
            frame = cv2.flip(frame, 1)
            # Keep consistent input aspect ratio; show the same transformed frame.
            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (640, int(h * 640 / w)))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            now = time.monotonic()
            ms = max(last_ms + 1, int(now * 1000))
            last_ms = ms
            result = detector.recognize_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ms)
            vector = None
            prediction = None
            svm_label = None
            dl_label, dl_score = 'no hand', 0.0
            if result.hand_landmarks:
                points = result.hand_landmarks[0]
                vector = features([(p.x,p.y,p.z) for p in points])
                for a,b in EDGES:
                    p,q = points[a], points[b]
                    cv2.line(frame, (int(p.x*frame.shape[1]),int(p.y*frame.shape[0])), (int(q.x*frame.shape[1]),int(q.y*frame.shape[0])), (155,230,90), 2)
                for p in points:
                    cv2.circle(frame,(int(p.x*frame.shape[1]),int(p.y*frame.shape[0])),4,(80,200,255),-1)
                if result.gestures and result.gestures[0]:
                    top = result.gestures[0][0]
                    dl_label, dl_score = top.category_name, top.score
                if model is not None:
                    svm_label = str(model.predict(vector.reshape(1,-1))[0])
                if mode == 'PRETRAINED DL':
                    prediction = GESTURES.get(dl_label) if dl_score >= 0.6 else None
                elif model is not None:
                    prediction = svm_label
            history.append((now,prediction))
            while history and history[0][0] < now - 0.7: history.popleft()
            votes = Counter(label for _,label in history if label)
            stable = None
            if votes and len(history) >= 3 and now - history[0][0] >= 0.35:
                candidate, count = votes.most_common(1)[0]
                if count / len(history) >= 0.75 and prediction == candidate: stable = candidate
            if capture:
                if now >= capture['ready'] and vector is not None and now >= capture['next']:
                    add_sample(DATA, session, capture['label'], vector)
                    capture['count'] += 1
                    capture['next'] = now + 0.2
                    if capture['count'] >= 40:
                        note = 'Saved 40 '+capture['label']+' samples. Collect other poses, then N.'
                        capture = None
                if capture and now - capture['ready'] > 30:
                    note = 'Capture timed out. Show your hand and press 1/2/3 again.'
                    capture = None
            if round_end and now >= round_end:
                if stable:
                    verdict = outcome(stable, computer)
                    scores[verdict] += 1
                    note = f'You: {stable.upper()} | CPU: {computer.upper()} | {verdict.upper()}! SPACE again.'
                else:
                    note = 'No stable move: round canceled, no score. Hold a pose; SPACE again.'
                round_end = None
            canvas = np.full((680,1100,3),(25,21,18),dtype=np.uint8)
            def text(s,x,y,size=0.6,color=(230,230,230),thick=1):
                cv2.putText(canvas,s,(x,y),cv2.FONT_HERSHEY_SIMPLEX,size,color,thick,cv2.LINE_AA)
            text('VISIONPLAY',28,44,1.1,(120,230,155),2)
            text('COMPUTER VISION / CLASSICAL ML / DEEP LEARNING',285,40,0.52,(180,180,180))
            display = cv2.resize(frame,(640,480))
            canvas[75:555,25:665] = display
            text(mode,700,100,0.7,(120,230,155),2)
            text('LIVE MOVE',700,145,0.5,(170,170,170))
            text((stable or 'hold a pose').upper(),700,182,0.85,(255,220,115),2)
            text(f'DL: {dl_label} ({dl_score:.2f})',700,215,0.52)
            text('DL score = confidence, not accuracy.',700,240,0.43,(170,170,170))
            text('SVM: '+(svm_label or ('no hand' if model is not None else 'not trained')),700,270,0.55)
            text(f'YOU {scores["win"]}  CPU {scores["lose"]}  DRAW {scores["draw"]}',700,310,0.6)
            text('SPACE   Play a round',700,355)
            text('M       Switch DL / SVM',700,385)
            text('1 / 2 / 3   Collect poses',700,415)
            text('N New session   T Train',700,445)
            text('E Evaluation   R Reset   Q Quit',700,475,0.49)
            text('1 rock   2 paper   3 scissors',700,515,0.48)
            if round_end:
                text(str(max(1,math.ceil(round_end-now))),310,325,3.0,(100,240,255),5)
                text('HOLD YOUR MOVE',220,370,0.7,(100,240,255),2)
            if capture:
                prep = max(0,math.ceil(capture['ready']-now))
                message = f'GET READY: {capture["label"]} ({prep})' if prep else f'CAPTURING {capture["label"]}: {capture["count"]}/40'
                text(message,30,590,0.67,(100,240,255),2)
            else:
                text('Session: '+session+' | SVM: '+('ready' if model is not None else 'collect 3 sessions to train'),30,590,0.55)
            # Wrap status rather than clip it.
            words, lines, line = note.split(), [], ''
            for word in words:
                if len(line+' '+word)>105: lines.append(line); line=''
                line += (' ' if line else '')+word
            lines.append(line)
            for i,line in enumerate(lines[:2]): text(line,30,625+i*25,0.53)
            elapsed = max(now-last_tick,1e-6)
            fps = 0.9*fps+0.1/elapsed
            last_tick = now
            text(f'{fps:.0f} FPS | local-only',850,655,0.45,(150,150,150))
            if show_eval:
                cv2.rectangle(canvas,(55,95),(650,545),(38,35,30),-1)
                text('YOUR SVM / SESSION HOLDOUT',80,135,0.75,(120,230,155),2)
                if report:
                    text(f'Accuracy: {report["accuracy"]:.1%}',80,185,0.8,(255,220,115),2)
                    text(f'Train: {report["train_samples"]}   Test: {report["test_samples"]} samples',80,225,0.6)
                    text('Confusion matrix: rows true, columns predicted',80,265,0.52)
                    text('                 rock   paper   scissors',80,300,0.54)
                    for i, row in enumerate(report['confusion_matrix_rows_true_columns_predicted']):
                        text(f'{LABELS[i]:10s}      {row[0]:4d}      {row[1]:4d}      {row[2]:4d}',80,335+i*32,0.55)
                    text('Small personal dataset - not general accuracy.',80,445,0.51)
                    text('Live SVM refit on all complete sessions.',80,475,0.51)
                else:
                    text('No evaluation yet. Collect 3 complete sessions,',80,205,0.55)
                    text('then press T. Each needs 1, 2 and 3 captures.',80,240,0.55)
                text('E to close | Full report: data/evaluation.json',80,520,0.5)
            cv2.imshow(window, canvas)
            key = cv2.waitKey(1) & 0xff
            if key in (ord('q'),27) or cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1: break
            if key == ord('e'): show_eval = not show_eval
            if key == ord('r'): scores.clear()
            if key == ord(' ') and not capture and not round_end:
                # CPU commits before it sees the player's final move.
                computer = secrets.choice(LABELS)
                round_end = now + 3
                history.clear()
                note = 'CPU has committed a hidden random move. Hold yours through the countdown.'
            if key == ord('m') and not capture and not round_end:
                if model is None: note = 'No trained SVM yet. Collect 3 sessions, then press T.'
                else:
                    mode = 'YOUR SVM' if mode == 'PRETRAINED DL' else 'PRETRAINED DL'
                    history.clear()
                    note = 'SVM always chooses one of 3 classes; it has no unknown-gesture rejection.' if mode == 'YOUR SVM' else 'Pretrained DL mode: fist / open palm / V sign.'
            if key in (ord('1'),ord('2'),ord('3')) and not round_end and not capture:
                capture = {'label': LABELS[key-ord('1')], 'ready': now+2, 'next':now+2, 'count':0}
                note = 'Pose label is YOUR keyboard choice, not the prediction. Gently vary position.'
            if key == ord('n') and not capture and not round_end:
                session = uuid.uuid4().hex[:12]
                note = 'New independent session. Change position/lighting slightly, collect all 3 poses.'
            if key == ord('t') and not capture and not round_end:
                try:
                    model, report = train(DATA, REPORT)
                    note = f'Held-out SESSION accuracy: {report["accuracy"]:.1%} on {report["test_samples"]} samples. Details: data/evaluation.json. M to use SVM.'
                except ValueError as exc: note = str(exc)
                history.clear()
    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera',type=int,default=0)
    parser.add_argument('--self-test',action='store_true')
    args = parser.parse_args()
    try:
        self_test() if args.self_test else main(args.camera)
    except Exception as exc:
        print('\nVisionPlay error:',exc)
        print('See README.md troubleshooting. If packages/model are missing, rerun setup.bat.')
        raise SystemExit(1)
