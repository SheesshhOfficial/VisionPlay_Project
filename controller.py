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
        self.running = True
        self.note = 'One hand, palm toward camera. First to 3 wins the match.'
    def action(self, action, now, svm_ready=False):
        if action == 'quit':
            self.game.cancel('Closed without scoring.'); self.wizard.cancel(); self.running=False
        elif action in ('training','teacher','back'):
            self.game.cancel('Round canceled on navigation; no score.')
            self.wizard.cancel()
            self.stability.clear()
            self.screen = 'play' if action=='back' else action
        elif action == 'play' and self.screen=='play':
            if self.game.start(now):
                self.stability.clear()
                self.note = 'Computer move locked before countdown. Hold your move steady.'
        elif action == 'reset' and self.screen=='play':
            self.game.reset(); self.stability.clear()
            self.note = 'New match. First to 3; draws do not count toward five.'
        elif action == 'mode' and self.screen=='play':
            if self.game.active: self.note='Finish the current round before switching classifiers.'
            elif not svm_ready: self.note='Train your SVM in Guided Training first.'
            else:
                self.mode='YOUR SVM' if self.mode=='PRETRAINED DL' else 'PRETRAINED DL'
                self.stability.clear()
                self.note='SVM chooses one of three classes; unfamiliar gestures can be misclassified.' if self.mode=='YOUR SVM' else 'Pretrained DL: fist, open palm or V sign.'
        elif action == 'capture' and self.screen=='training': self.wizard.begin(now)
        elif action == 'cancel' and self.screen=='training': self.wizard.cancel()
        elif action == 'train' and self.screen=='training' and not self.wizard.active:
            return 'train'
        return None
    def camera_lost(self):
        self.game.cancel('Camera disconnected; round canceled without score.')
        self.wizard.cancel(); self.stability.clear(); self.running=False
