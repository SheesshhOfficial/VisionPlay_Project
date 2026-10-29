"""Dependency-light OpenCV presentation layer; no camera or model side effects."""
import math
import cv2
import numpy as np

BG=(25,20,17); CARD=(44,36,29); TEXT=(238,239,235); MUTED=(175,169,157)
GREEN=(160,225,99); GOLD=(97,203,255); BLUE=(240,181,101)

def render(frame, state):
    """Return a 1280x800 BGR frame and clickable action -> (x,y,w,h)."""
    out=np.full((800,1280,3),BG,dtype=np.uint8)
    buttons={}
    def label(s,x,y,size=.55,color=TEXT,bold=1):
        cv2.putText(out,str(s),(x,y),cv2.FONT_HERSHEY_SIMPLEX,size,color,bold,cv2.LINE_AA)
    def wrap(s,x,y,width=72,size=.52,color=MUTED,lines=3):
        line=''; result=[]
        for word in str(s).split():
            if len(line+' '+word)>width: result.append(line);line=''
            line+=(' ' if line else '')+word
        result.append(line)
        for i,line in enumerate(result[:lines]): label(line,x,y+i*24,size,color)
    def panel(x,y,w,h): cv2.rectangle(out,(x,y),(x+w,y+h),CARD,-1)
    def button(action,title,x,y,w=150,color=GREEN):
        cv2.rectangle(out,(x,y),(x+w,y+44),color,-1)
        size=min(.54,(w-18)/max(1,len(title)*11.5))
        label(title,x+10,y+28,size,BG,1)
        buttons[action]=(x,y,w,44)
    def icon(move,cx,cy,scale=1,color=GREEN):
        """Stylized vector hand silhouettes, drawn locally rather than image assets."""
        def rect(x1,y1,x2,y2):
            cv2.rectangle(out,(int(cx+x1*scale),int(cy+y1*scale)),(int(cx+x2*scale),int(cy+y2*scale)),color,-1)
        def line(x1,y1,x2,y2,t=17):
            a=(int(cx+x1*scale),int(cy+y1*scale));b=(int(cx+x2*scale),int(cy+y2*scale))
            cv2.line(out,a,b,color,max(1,int(t*scale)),cv2.LINE_AA)
            cv2.circle(out,a,max(1,int(t*scale/2)),color,-1);cv2.circle(out,b,max(1,int(t*scale/2)),color,-1)
        if move in ('rock','paper','scissors'):
            rect(-33,-4,29,49);rect(-19,42,19,67)
            if move=='rock':
                for x in (-25,-9,7,23): line(x,-18,x,10,17)
                line(-34,11,-13,27,18)
            elif move=='paper':
                for x,top in [(-25,-43),(-9,-65),(7,-70),(23,-53)]: line(x,3,x,top,13)
                line(-29,25,-49,-6,15)
            else:
                line(-14,3,-32,-65,17);line(6,3,26,-68,17)
                line(23,4,24,25,16);line(-30,16,-8,32,17)
        else:
            cv2.ellipse(out,(cx,cy-14),(27,29),0,180,360,MUTED,8,cv2.LINE_AA)
            rect(-36,-13,36,46)
            cv2.circle(out,(cx,cy+10),7,BG,-1)
            cv2.line(out,(cx,cy+10),(cx,cy+26),BG,5)
    label('VISIONPLAY',24,45,1.05,GREEN,2)
    label('PLAY  /  LEARN  /  EXPLAIN',300,42,.55,MUTED)
    label(f'{state.get("fps",0):.0f} FPS  |  CAMERA STAYS LOCAL',955,40,.45,MUTED)
    screen=state.get('screen','play'); game=state.get('game',{}); tr=state.get('training',{})
    if screen in ('play','training'):
        panel(20,80,650,510)
        if frame is not None and getattr(frame,'size',0):
            h,w=frame.shape[:2]; factor=min(640/w,480/h)
            image=cv2.resize(frame,(max(1,int(w*factor)),max(1,int(h*factor))))
            yy=100+(480-image.shape[0])//2;xx=25+(640-image.shape[1])//2
            out[yy:yy+image.shape[0],xx:xx+image.shape[1]]=image
        else: label('Waiting for camera...',175,330,.8,MUTED)
        label('LIVE CAMERA',35,99,.42,MUTED)
        progress=max(0.,min(1.,float(state.get('stability',0))))
        cv2.rectangle(out,(25,607),(665,618),CARD,-1)
        if progress: cv2.rectangle(out,(25,607),(25+int(640*progress),618),GREEN,-1)
        label('STABILITY',25,642,.43,MUTED)
        wrap(state.get('guidance','Show one hand and hold your pose.'),135,642,65,.48,lines=1)
    if screen=='play':
        panel(690,80,570,510)
        label('BEST OF FIVE',714,115,.7,GOLD,2)
        label('First to 3 wins. Draws do not advance the match.',714,144,.47,MUTED)
        pw=int(game.get('player_wins',0));cw=int(game.get('cpu_wins',0))
        label('YOU',760,187,.55);label('COMPUTER',1053,187,.55)
        label(f'{pw}  :  {cw}',905,190,.95,GREEN,2)
        phase=game.get('phase','ready')
        pm=game.get('player_move') or state.get('live_move')
        cm=game.get('computer_move') if phase in ('result','match_over') else None
        icon(pm,795,292,.86,GREEN);icon(cm,1124,292,.86,BLUE)
        label(str(pm or 'YOUR MOVE').upper(),739,383,.6,GREEN,2)
        label(str(cm or ('LOCKED' if phase in ('countdown','grace') else 'HIDDEN')).upper(),1060,383,.6,BLUE,2)
        if phase in ('countdown','grace'):
            if phase=='countdown':
                remaining=float(game.get('countdown',3));n=max(1,math.ceil(remaining))
                radius=35+int(7*(1-(remaining%1)))
                cv2.circle(out,(965,285),radius,GOLD,2,cv2.LINE_AA)
                label(n,949,299,1.2,GOLD,2)
            else: label('HOLD',925,285,.8,GOLD,2)
            label('Computer move locked before countdown.',720,430,.5,MUTED)
            wrap('Hold your pose a moment longer.' if phase=='grace' else 'Show your move. We reveal together.',720,466,54,.6,GOLD)
        else:
            result=game.get('result_text') or 'Ready when you are. Start a round!'
            wrap(result,720,432,50,.65,GREEN if game.get('winner')=='player' else GOLD,3)
        label(f'Draws: {game.get("draws",0)}  |  CPU chooses randomly, not adaptively.',714,570,.43,MUTED)
        label('GAME MODE: '+str(state.get('mode','PRETRAINED DL')),25,682,.52,GREEN)
        label(f'DL: {state.get("dl_label","no hand")}  score {state.get("dl_score",0):.2f}',25,710,.48)
        label('SVM: '+str(state.get('svm_label') or ('no hand' if state.get('svm_ready') else 'not trained')),360,710,.48)
        label('NEURAL: '+str(state.get('neural_label') or ('no hand' if state.get('neural_ready') else 'not trained'))+f'  score {state.get("neural_score",0):.2f}',700,602,.43)
        label('Neural score = model output, not accuracy.',700,624,.47,MUTED)
        button('play','SPACE  Start round',700,651,190)
        button('reset','R  New match',905,651,160,BLUE)
        button('mode','M  Switch mode',1080,651,180,BLUE)
        button('training','Training wizard',700,709,190,GOLD)
        button('teacher','Teacher screen',905,709,180,GOLD)
        button('quit','Q  Quit',1100,709,160,MUTED)
    elif screen=='training':
        panel(690,80,570,560)
        label('TRAIN YOUR CLASSIFIERS',712,119,.75,GREEN,2)
        wrap('Capture real examples. Every session includes all three poses.',712,154,57,.5,MUTED,2)
        label(f'Complete sessions saved: {tr.get("complete_sessions",0)} / 3 minimum',712,221,.56)
        label('Neural MLP: '+str(state.get('neural_status','not trained')),712,242,.43,MUTED)
        label(f'CURRENT SESSION {tr.get("session_index",1)}',712,263,.5,GOLD)
        move=tr.get('label','rock');count=int(tr.get('count',0))
        icon(move,800,367,.78,GREEN)
        label(str(move).upper(),912,334,.85,GREEN,2)
        label(f'{count} / 40 samples',912,377,.64)
        if float(tr.get('ready_in',0)) > 0:
            label(f'Get ready: {math.ceil(float(tr["ready_in"]))}',912,449,.55,GOLD)
        cv2.rectangle(out,(912,402),(1230,415),BG,-1)
        cv2.rectangle(out,(912,402),(912+int(318*min(1,count/40)),415),GREEN,-1)
        wrap(tr.get('status','Choose capture, then hold the requested pose.'),712,478,56,.55,TEXT,3)
        wrap('The label is your instruction, not the AI prediction. Change position between sessions.',712,574,65,.45,MUTED,2)
        button('capture','SPACE  Capture / Next',25,674,230)
        button('train','T  Train + evaluate',275,674,220,GOLD)
        button('train_neural','N  Train neural net',505,674,210,BLUE)
        button('cancel','Cancel capture',725,674,175,MUTED)
        button('back','Back to game',910,674,180,BLUE)
        label('Samples are saved locally. Cancel keeps already captured rows.',25,746,.48,MUTED)
    else:
        label('FROM CAMERA TO CLASSROOM',25,98,.85,GOLD,2)
        steps=[('01 / COMPUTER VISION','OpenCV reads camera frames.','Color conversion + hand overlay.'),('02 / PRETRAINED DL','Google neural models find a hand,','21 landmarks and gesture labels.'),('03 / YOUR MLP + SVM','Train on pretrained model','landmark features.'),('04 / GAME LOGIC','A fixed random CPU move,','countdown, rules and scoreboard.')]
        for i,(title,a,b) in enumerate(steps):
            x=25+i*315;panel(x,125,300,135)
            label(title,x+13,155,.47,GREEN,1)
            label(a,x+13,196,.43);label(b,x+13,222,.43)
        panel(25,285,770,395);panel(815,285,440,395)
        neural=state.get('neural_evaluation')
        show_neural=state.get('teacher_model')=='NEURAL NET'
        report=neural if show_neural else state.get('evaluation')
        kind='NEURAL NET' if show_neural else 'SVM'
        label(f'REAL HELD-OUT {kind} EVALUATION',45,325,.65,GREEN,2)
        if report:
            label(f'{kind} accuracy: {report.get("accuracy",0):.1%}',45,373,.85,GOLD,2)
            label(f'Train {report.get("train_samples",0)}  |  Test {report.get("test_samples",0)} samples',45,411,.56)
            label('Rows = true class. Columns = predicted class.',45,448,.5,MUTED)
            for text,xpos in [('ROCK',246),('PAPER',374),('SCISSORS',500)]: label(text,xpos,486,.5,GREEN)
            for i,row in enumerate(report.get('confusion_matrix_rows_true_columns_predicted',[])):
                if i>2: break
                label(('ROCK','PAPER','SCISSORS')[i],45,524+i*35,.52)
                for val,xpos in zip(row,(246,374,500)): label(f'{val:4d}',xpos,524+i*35,.52)
            label('Small personal dataset, not universal performance.',45,655,.46,MUTED)
        else:
            wrap('No measured model results yet. Open the training wizard, capture three complete sessions, then train.',45,388,65,.65,TEXT,4)
            label('No invented accuracy. No fake trained model.',45,595,.57,GOLD)
        label('EXPLAIN IT HONESTLY',836,325,.61,GOLD,2)
        wrap('The pretrained model extracts landmarks. Your MLP trains on them, not raw-image CNN input.',836,368,43,.5,TEXT,3)
        wrap('Evaluation holds out an entire session; nearby frames are correlated.',836,442,43,.48,TEXT,3)
        if show_neural and report and report.get('training_loss_curve', report.get('loss_curve')):
            values=[float(v) for v in report.get('training_loss_curve',report.get('loss_curve',[])) if math.isfinite(float(v))]
            x0,y0,ww,hh=865,520,330,100
            cv2.rectangle(out,(x0,y0),(x0+ww,y0+hh),BG,-1)
            if len(values)>1:
                lo,hi=min(values),max(values); span=max(hi-lo,1e-9)
                pts=[(x0+int(i*ww/(len(values)-1)),y0+hh-int((v-lo)/span*hh)) for i,v in enumerate(values)]
                for a,b in zip(pts,pts[1:]): cv2.line(out,a,b,GOLD,2,cv2.LINE_AA)
                label(f'{hi:.2f}',x0-35,y0+9,.37,MUTED);label(f'{lo:.2f}',x0-35,y0+hh,.37,MUTED)
            label('TRAINING LOSS (lower is better)',836,506,.42,GREEN)
            label('epoch ->',1135,645,.40,MUTED)
        elif not show_neural:
            wrap('The live SVM is refit on all complete sessions. Its training score is not independent.',836,523,43,.48,MUTED,4)
        button('back','Back to game',25,705,200)
        button('training','Open training wizard',245,705,240,GOLD)
        label('V switches report | data/*evaluation.json',515,733,.47,MUTED)
        button('teacher_toggle','V  Switch report',1010,705,225,BLUE)
    # Status strip stays separate from controls and uses one concise line.
    wrap(state.get('note',''),25,786,145,.43,MUTED,1)
    return out, buttons
