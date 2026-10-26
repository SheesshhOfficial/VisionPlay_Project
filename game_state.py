"""Deterministic, camera-independent round/match state.

Best of five means first to three decisive wins. Draws and canceled rounds
never consume one of the five decisive rounds. The random opponent commits
once at start, before the final player gesture is observed.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional
import secrets
from core import LABELS, outcome


@dataclass
class RoundGame:
    choose: Callable = field(default=secrets.choice, repr=False)
    countdown_seconds: float = 3.0
    grace_seconds: float = 1.5
    phase: str = field(default='ready', init=False)
    player_wins: int = field(default=0, init=False)
    cpu_wins: int = field(default=0, init=False)
    draws: int = field(default=0, init=False)
    rounds_played: int = field(default=0, init=False)
    player_move: Optional[str] = field(default=None, init=False)
    result: Optional[str] = field(default=None, init=False)
    result_text: str = field(default='Ready for a best-of-five match.', init=False)
    _computer: Optional[str] = field(default=None, init=False, repr=False)
    _started: Optional[float] = field(default=None, init=False, repr=False)

    @property
    def active(self):
        return self.phase in ('countdown', 'grace')

    @property
    def decisive_rounds(self):
        return self.player_wins + self.cpu_wins

    @property
    def winner(self):
        if self.player_wins >= 3:
            return 'player'
        if self.cpu_wins >= 3:
            return 'computer'
        return None

    @property
    def computer_move(self):
        """Reveal only after a valid round resolves; never during countdown."""
        if self.phase in ('result', 'match_over') and self.result in ('win', 'lose', 'draw'):
            return self._computer
        return None

    def countdown_remaining(self, now):
        if self._started is None or not self.active:
            return 0.0
        return max(0.0, self._started + self.countdown_seconds - now)

    def grace_remaining(self, now):
        if self._started is None or not self.active:
            return 0.0
        return max(0.0, self._started + self.countdown_seconds + self.grace_seconds - now)

    def start(self, now):
        """Repeated active/completed-match clicks are harmless and consume no RNG."""
        if self.active or self.winner:
            return False
        computer = self.choose(LABELS)
        if computer not in LABELS:
            raise ValueError('Opponent chooser returned an invalid move')
        self._computer = computer
        self._started = now
        self.player_move = None
        self.result = None
        self.result_text = 'Computer move locked. Hold your move steady.'
        self.phase = 'countdown'
        return True

    def tick(self, now, stable_move):
        """Advance only from a stable valid gesture, with a bounded grace window.

        A late frame after the grace deadline cancels even if it has a gesture:
        it must not retroactively count a move never observed before the deadline.
        """
        if not self.active:
            return False
        countdown_end = self._started + self.countdown_seconds
        deadline = countdown_end + self.grace_seconds
        if now < countdown_end:
            return False
        if now > deadline:
            return self.cancel('No stable move in time. Round canceled; no score.')
        if stable_move in LABELS:
            self.player_move = stable_move
            self.result = outcome(stable_move, self._computer)
            self.rounds_played += 1
            if self.result == 'win':
                self.player_wins += 1
                self.result_text = 'You win this round!'
            elif self.result == 'lose':
                self.cpu_wins += 1
                self.result_text = 'Computer wins this round.'
            else:
                self.draws += 1
                self.result_text = 'Draw! Replay this decisive round.'
            self.phase = 'match_over' if self.winner else 'result'
            if self.winner:
                self.result_text = ('You win the match!' if self.winner == 'player' else 'Computer wins the match.') + ' New match to play again.'
            return True
        if now >= deadline:
            return self.cancel('No stable move in time. Round canceled; no score.')
        if self.phase != 'grace':
            self.phase = 'grace'
            self.result_text = 'One extra moment: hold a clear move steady.'
            return True
        return False

    def cancel(self, reason='Round canceled; no score.'):
        """Cancel an active round on navigation/interruption without scoring."""
        if not self.active:
            return False
        self.phase = 'result'
        self.result = 'cancel'
        self.result_text = reason
        self.player_move = None
        self._computer = None
        return True

    def reset(self):
        self.phase = 'ready'
        self.player_wins = self.cpu_wins = self.draws = self.rounds_played = 0
        self.player_move = self.result = self._computer = self._started = None
        self.result_text = 'Ready for a best-of-five match.'
