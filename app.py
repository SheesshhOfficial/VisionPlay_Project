"""VisionPlay: local webcam rock-paper-scissors with DL and trainable SVM."""
from pathlib import Path
import argparse
import time
import sys
import numpy as np
from core import LABELS, GESTURES, features, outcome
from learning import train
from controller import Controller
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
    from ui import render
    state = Controller(DATA)
    detector = recognizer()
    cap = None
    window = 'VisionPlay | CV + Machine Learning + Deep Learning'
    model = report = None
    last_ms = 0
    last_tick = time.monotonic()
    fps = 0.0
    buttons = {}
    pending_actions = []
    def mouse(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONUP:
            for action,(bx,by,bw,bh) in buttons.items():
                if bx<=x<bx+bw and by<=y<by+bh:
                    # One action per displayed frame prevents double-click state skipping.
                    if not pending_actions: pending_actions.append(action)
                    break
    try:
        cap = cv2.VideoCapture(camera,cv2.CAP_DSHOW) if sys.platform=='win32' else cv2.VideoCapture(camera)
        if not cap.isOpened():
            cap.release(); cap=cv2.VideoCapture(camera)
        if not cap.isOpened(): raise RuntimeError('Cannot open webcam. Close camera apps; try runme.bat 1.')
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        cv2.namedWindow(window,cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window,1120,700)
        cv2.setMouseCallback(window,mouse)
        if DATA.exists():
            try: model,report=train(DATA,REPORT)
            except ValueError as exc: state.note='Saved samples preserved. '+str(exc)
        while state.running:
            ok,frame=cap.read()
            if not ok:
                state.camera_lost()
                raise RuntimeError('Webcam disconnected. Round canceled; saved samples kept. Reconnect and restart.')
            frame=cv2.flip(frame,1)
            h,w=frame.shape[:2]
            frame=cv2.resize(frame,(640,int(h*640/w)))
            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            now=time.monotonic()
            last_ms=max(last_ms+1,int(now*1000))
            result=detector.recognize_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb),last_ms)
            vector=prediction=svm_label=None
            dl_label,dl_score='no hand',0.0
            guidance='Show one hand fully inside the camera view.'
            if result.hand_landmarks:
                points=result.hand_landmarks[0]
                vector=features([(p.x,p.y,p.z) for p in points])
                for a,b in EDGES:
                    p,q=points[a],points[b]
                    cv2.line(frame,(int(p.x*frame.shape[1]),int(p.y*frame.shape[0])),(int(q.x*frame.shape[1]),int(q.y*frame.shape[0])),(155,230,90),2)
                for p in points: cv2.circle(frame,(int(p.x*frame.shape[1]),int(p.y*frame.shape[0])),4,(80,200,255),-1)
                if result.gestures and result.gestures[0]:
                    top=result.gestures[0][0]; dl_label,dl_score=top.category_name,top.score
                if model is not None: svm_label=str(model.predict(vector.reshape(1,-1))[0])
                prediction=(GESTURES.get(dl_label) if dl_score>=0.6 else None) if state.mode=='PRETRAINED DL' else svm_label
                extent=max(max(p.x for p in points)-min(p.x for p in points),max(p.y for p in points)-min(p.y for p in points))
                edge=any(p.x<0.02 or p.x>0.98 or p.y<0.02 or p.y>0.98 for p in points)
                if edge: guidance='Keep the whole hand inside the frame.'
                elif extent<0.18: guidance='Hand looks small in frame. Try moving a little closer.'
                elif prediction is None: guidance='Try a clear fist, open palm or V sign; face the camera.'
                else: guidance='Hold this pose steady through the countdown.'
            stable,progress=state.stability.update(now,prediction)
            if stable: guidance='Stable move ready. Keep holding it.'
            state.game.tick(now,stable)
            state.wizard.tick(now,vector)
            elapsed=max(now-last_tick,1e-6);fps=0.9*fps+0.1/elapsed;last_tick=now
            game=state.game; wizard=state.wizard
            view={
                'screen':state.screen,'mode':state.mode,'dl_label':dl_label,'dl_score':dl_score,
                'svm_label':svm_label,'svm_ready':model is not None,'fps':fps,'note':state.note,
                'stability':progress,'guidance':guidance,'evaluation':report,
                'game':{'phase':game.phase,'countdown':game.countdown_remaining(now),'grace':game.grace_remaining(now),
                        'player_move':game.player_move or stable,'computer_move':game.computer_move,
                        'player_wins':game.player_wins,'cpu_wins':game.cpu_wins,'draws':game.draws,
                        'result_text':game.result_text,'winner':game.winner},
                'training':{'status':wizard.status,'label':wizard.label,'session_index':wizard.session_number,
                            'count':wizard.count,'active':wizard.active,'complete_sessions':wizard.completed_sessions,
                            'phase':wizard.phase,'counts':wizard.counts,'ready_in':max(0,wizard.ready_at-now)}
            }
            canvas,buttons=render(frame,view)
            cv2.imshow(window,canvas)
            key=cv2.waitKey(1)&0xff
            window_closed=cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1
            keys={ord('q'):'quit',ord('g'):'training',ord('e'):'teacher',ord('m'):'mode',ord('r'):'reset',ord('t'):'train',ord('c'):'capture',27:'back'}
            if key==ord(' '): action='capture' if state.screen=='training' else 'play'
            else: action=keys.get(key)
            if action: pending_actions[:]=[action]
            if window_closed: pending_actions[:]=['quit']
            if pending_actions:
                action=pending_actions.pop(0)
                command=state.action(action,now,model is not None)
                if command=='train':
                    try:
                        model,report=train(DATA,REPORT)
                        state.note=f'SVM trained. Held-out session: {report["accuracy"]:.1%} on {report["test_samples"]} samples. Teacher View shows details.'
                    except ValueError as exc: state.note=str(exc)
                    state.stability.clear()
    finally:
        state.wizard.cancel()
        if cap is not None: cap.release()
        detector.close()
        cv2.destroyAllWindows()

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera',type=int,default=0)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    try: self_test() if args.self_test else main(args.camera)
    except Exception as exc:
        print('\nVisionPlay error:',exc)
        print('See README.md. If packages/model are missing, rerun setup.bat.')
        raise SystemExit(1)
