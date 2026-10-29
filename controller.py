"""Testable UI navigation, prediction stability, and camera-loss safety."""
from collections import Counter, deque
from game_state import RoundGame
from training_state import TrainingWizard

class Stability:
    def __init__(self): self.history = deque()
    def clear(self): self.history.clear()
    def update(self, now, prediction):
        self.history.append((now,prediction))
        while self.history and self.history[0][0] < now - 0.7: self.history.popleft()
        votes = Counter(p for _,p in self.history if p)
        if not votes: return None, 0.0
        candidate, count = votes.most_common(1)[0]
        duration = now-self.history[0][0]
        fraction = count/len(self.history)
        progress = min(1.0,duration/0.35)*min(1.0,fraction/0.75)*min(1.0,len(self.history)/3)
        stable = candidate if len(self.history)>=3 and duration>=0.35 and fraction>=0.75 and prediction==candidate else None
        return stable, progress if prediction==candidate else 0.0

class Controller:
    def __init__(self, data_path):
        self.game = RoundGame()
        self.wizard = TrainingWizard(data_path)
        self.stability = Stability()
        self.screen = 'play'
        self.mode = 'PRETRAINED DL'
        self.available_modes = ['PRETRAINED DL']
        self.training_job = None
        self.teacher_model = 'SVM'
        self.running = True
        self.note = 'One hand, palm toward camera. First to 3 wins the match.'
    def action(self, action, now, svm_ready=False, neural_ready=False):
        if action == 'quit':
            self.game.cancel('Closed without scoring.'); self.wizard.cancel(); self.running=False
        elif action in ('training','teacher','back'):
            self.game.cancel('Round canceled on navigation; no score.')
            self.wizard.cancel()
            self.stability.clear()
            self.screen = 'play' if action=='back' else action
        elif action == 'play' and self.screen=='play':
            if self.training_job:
                self.note='Wait for background training to finish before starting a round.'
                return
            if self.game.start(now):
                self.stability.clear()
                self.note = 'Computer move locked before countdown. Hold your move steady.'
        elif action == 'reset' and self.screen=='play':
            self.game.reset(); self.stability.clear()
            self.note = 'New match. First to 3; draws do not count toward five.'
        elif action == 'mode' and self.screen=='play':
            if self.training_job: self.note='Wait for background training to finish before switching classifiers.'
            elif self.game.active: self.note='Finish the current round before switching classifiers.'
            else:
                modes=['PRETRAINED DL'] + (['YOUR SVM'] if svm_ready else []) + (['YOUR NEURAL NET'] if neural_ready else [])
                if len(modes)==1: self.note='Train a model in Guided Training first.'; return
                self.mode=modes[(modes.index(self.mode)+1)%len(modes)] if self.mode in modes else modes[0]
                self.stability.clear()
                self.note=('Your neural net classifies DL-extracted hand landmarks; unfamiliar poses may be misclassified.' if self.mode=='YOUR NEURAL NET' else 'SVM chooses one of three classes; unfamiliar gestures can be misclassified.' if self.mode=='YOUR SVM' else 'Pretrained DL: fist, open palm or V sign.')
        elif action == 'capture' and self.screen=='training':
            if self.training_job: self.note='Wait for training to finish before capturing more samples.'
            else: self.wizard.begin(now)
        elif action == 'teacher_toggle' and self.screen=='teacher': self.teacher_model='NEURAL NET' if self.teacher_model=='SVM' else 'SVM'
        elif action == 'cancel' and self.screen=='training': self.wizard.cancel()
        elif action == 'train' and self.screen=='training' and not self.wizard.active:
            if self.training_job: self.note='A model is already training.'
            else: self.training_job='svm'; return 'train'
        elif action == 'train_neural' and self.screen=='training' and not self.wizard.active:
            if self.training_job: self.note='A model is already training.'
            elif self.wizard.completed_sessions < 3: self.note='Need 3 complete sessions before neural training.'
            else: self.training_job='neural'; return 'train_neural'
        return None
    def training_finished(self): self.training_job=None
    def camera_lost(self):
        self.game.cancel('Camera disconnected; round canceled without score.')
        self.wizard.cancel(); self.stability.clear(); self.running=False
