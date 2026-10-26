"""Pure game state tests; no webcam accuracy claims."""
import sys
from pathlib import Path
import unittest
from unittest.mock import Mock
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from game_state import RoundGame


class RoundGameTests(unittest.TestCase):
    def make_game(self, move='rock'):
        chooser = Mock(return_value=move)
        return RoundGame(choose=chooser), chooser

    def test_commit_hidden_until_resolved_and_repeat_start(self):
        game, chooser = self.make_game()
        self.assertTrue(game.start(10))
        self.assertFalse(game.start(11))
        self.assertIsNone(game.computer_move)
        self.assertFalse(game.tick(12.9, 'paper'))
        self.assertEqual(chooser.call_count, 1)
        self.assertTrue(game.tick(13, 'paper'))
        self.assertEqual(game.computer_move, 'rock')
        self.assertEqual(game.result, 'win')
        self.assertEqual(game.player_wins, 1)
        self.assertFalse(game.tick(14, 'scissors'))
        self.assertEqual(game.player_wins, 1)

    def test_grace_then_stable(self):
        game, chooser = self.make_game()
        game.start(0)
        self.assertTrue(game.tick(3, None))
        self.assertEqual(game.phase, 'grace')
        self.assertIsNone(game.computer_move)
        self.assertAlmostEqual(game.grace_remaining(3.5), 1)
        self.assertTrue(game.tick(4.4, 'paper'))
        self.assertEqual(game.player_wins, 1)
        self.assertEqual(chooser.call_count, 1)

    def test_grace_expiry_without_move(self):
        game, _ = self.make_game()
        game.start(0)
        game.tick(3, None)
        self.assertFalse(game.tick(4.49, None))
        self.assertTrue(game.tick(4.5, None))
        self.assertEqual(game.result, 'cancel')
        self.assertIsNone(game.computer_move)
        self.assertEqual(game.rounds_played, 0)
        self.assertTrue(game.start(6))

    def test_late_frame_does_not_retroactively_score(self):
        game, _ = self.make_game()
        game.start(0)
        game.tick(4.51, 'paper')
        self.assertEqual(game.result, 'cancel')
        self.assertEqual(game.player_wins, 0)

    def test_deadline_stable_move_accepted(self):
        game, _ = self.make_game()
        game.start(0)
        game.tick(4.5, 'paper')
        self.assertEqual(game.result, 'win')

    def test_ties_do_not_advance_best_of_five(self):
        game, _ = self.make_game()
        for index in range(6):
            game.start(index * 10)
            game.tick(index * 10 + 3, 'rock')
        self.assertEqual(game.draws, 6)
        self.assertEqual(game.decisive_rounds, 0)
        self.assertIsNone(game.winner)
        for index in range(3):
            game.start(100 + index * 10)
            game.tick(103 + index * 10, 'paper')
        self.assertEqual(game.winner, 'player')
        self.assertEqual(game.phase, 'match_over')
        self.assertEqual(game.rounds_played, 9)
        self.assertFalse(game.start(200))
        game.reset()
        self.assertEqual(game.phase, 'ready')
        self.assertEqual(game.rounds_played, 0)
        self.assertIsNone(game.computer_move)
        self.assertTrue(game.start(201))

    def test_cpu_match_win_after_five_decisive_rounds(self):
        game, _ = self.make_game()
        for index, move in enumerate(['scissors', 'paper', 'scissors', 'paper', 'scissors']):
            game.start(index * 10)
            game.tick(index * 10 + 3, move)
        self.assertEqual(game.winner, 'computer')
        self.assertEqual(game.decisive_rounds, 5)
        self.assertEqual((game.player_wins, game.cpu_wins), (2, 3))

    def test_navigation_or_camera_loss_cancels_once(self):
        game, _ = self.make_game()
        game.start(0)
        self.assertTrue(game.cancel('Camera lost'))
        self.assertFalse(game.cancel('Back'))
        self.assertEqual(game.result_text, 'Camera lost')
        self.assertEqual(game.decisive_rounds, 0)
        self.assertEqual(game.rounds_played, 0)
        self.assertFalse(game.tick(3, 'paper'))

    def test_unknown_move_cannot_score(self):
        game, _ = self.make_game()
        game.start(0)
        game.tick(3, 'thumbs_up')
        self.assertEqual(game.phase, 'grace')
        game.tick(4.5, 'thumbs_up')
        self.assertEqual(game.result, 'cancel')


if __name__ == '__main__':
    unittest.main()
