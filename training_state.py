"""Guided, interruptible local landmark capture; no predictions used as labels."""
from collections import Counter
from pathlib import Path
import uuid
import numpy as np
from core import LABELS
from learning import add_sample, read_samples


class TrainingWizard:
    TARGET = 40
    PREPARATION_SECONDS = 2.0
    SAMPLE_INTERVAL = 0.2
    CAPTURE_TIMEOUT = 30.0

    def __init__(self, csv_path):
        self.csv_path = Path(csv_path)
        self.completed_sessions = 0
        self._prior_complete = 0
        if self.csv_path.exists() and self.csv_path.stat().st_size:
            _, labels, sessions = read_samples(self.csv_path)
            counts = Counter(zip(sessions, labels))
            # Match the unchanged evaluator's eligibility rule for legacy data.
            self._prior_complete = sum(all(counts[s, label] >= 20 for label in LABELS)
                                       for s in set(sessions))
        self.completed_sessions = self._prior_complete
        self._new_session()
        self.phase = 'ready'
        self.status = 'Show ROCK, then Start capture. Labels come from your chosen pose.'

    def _new_session(self):
        self.session_id = uuid.uuid4().hex[:12]
        self.counts = {label: 0 for label in LABELS}
        self.label_index = 0
        self.ready_at = 0.0
        self.next_sample_at = 0.0
        self.session_number = self.completed_sessions + 1
        self._counted_current = False

    @property
    def label(self):
        return LABELS[self.label_index]

    @property
    def count(self):
        return self.counts[self.label]

    @property
    def active(self):
        return self.phase in ('preparing', 'capturing')

    @property
    def training_ready(self):
        return self.completed_sessions >= 3

    def begin(self, now):
        """Explicit Start/Next. Complete sessions can be extended for more data."""
        if self.active:
            return False
        if self.phase == 'batch_done':
            self.label_index += 1
        elif self.phase in ('session_done', 'complete'):
            self._new_session()
        self.ready_at = now + self.PREPARATION_SECONDS
        self.next_sample_at = self.ready_at
        self.phase = 'preparing'
        self.status = f'Get ready: {self.label.upper()}. Keep this pose for the whole batch.'
        return True

    def cancel(self):
        """Pause on Back/Close/camera loss without deleting any collected rows."""
        if not self.active:
            return False
        self.phase = 'ready'
        self.status = f'Paused {self.label}: {self.count}/40 saved. Start resumes this batch.'
        return True

    def tick(self, now, vector):
        """At most one observed real frame per interval; never synthesize catch-up frames."""
        if not self.active:
            return
        if now < self.ready_at:
            return
        if now - self.ready_at > self.CAPTURE_TIMEOUT:
            self.cancel()
            self.status = f'Capture paused after 30 seconds. {self.count}/40 saved; show hand and resume.'
            return
        self.phase = 'capturing'
        if vector is None:
            self.status = f'Waiting for a visible hand. {self.label}: {self.count}/40 saved.'
            return
        if now < self.next_sample_at:
            return
        values = np.asarray(vector, dtype=float)
        if values.shape != (63,) or not np.isfinite(values).all():
            raise ValueError('Capture requires 63 finite landmark features.')
        add_sample(self.csv_path, self.session_id, self.label, values)
        self.counts[self.label] += 1
        self.next_sample_at = now + self.SAMPLE_INTERVAL
        self.status = f'Capturing {self.label.upper()}: {self.count}/40. Keep the same pose.'
        if self.count >= self.TARGET:
            if self.label_index < len(LABELS) - 1:
                self.phase = 'batch_done'
                self.status = f'{self.label.upper()} saved. Next pose: {LABELS[self.label_index+1].upper()}. Click Next when ready.'
            else:
                if not self._counted_current:
                    self.completed_sessions += 1
                    self._counted_current = True
                self.phase = 'complete' if self.training_ready else 'session_done'
                self.status = ('Three complete sessions ready. Train to measure held-out results; Next adds another session.'
                               if self.training_ready else
                               'Session complete. Reposition your hand/change lighting slightly, then Next session.')
