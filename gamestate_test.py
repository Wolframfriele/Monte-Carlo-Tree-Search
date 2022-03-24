"""Unit tests for the monte carlo tree search library."""
import unittest
from mcts_tic_tac_toe import GameState, GameSimulator


class GameStateTestCase(unittest.TestCase):
    """Test basic gamestate"""
    def test_is_won(self):
        """Checks if the is_won function returns correct results"""
        input_positions = [(0b000000000, 0b000000000),
                           (0b010100010, 0b111000000),
                           (0b111000000, 0b000110000),
                           (0b000111000, 0b000000000),
                           (0b000000111, 0b011010000),
                           (0b100100100, 0b001010100)]

        expected = [0, 1, 1 ,1 ,1 ,1]
        result = []

        for position in input_positions:
            board = GameState({'player': False, 'board': position})
            result.append(board.is_won())
        self.assertEqual(result, expected)

class GameSimulatorTestCase(unittest.TestCase):
    """Tests the GameSimulator board"""
    def test_is_won_player_0(self):
        """Sets up a almost terminal state that player 0 has to win"""
        game = GameSimulator({'player': True, 'board': (0b100001110, 0b011010001)})
        self.assertEqual(game.play_out(), 0)

    def test_is_won_player_1(self):
        """Sets up a almost terminal state that player 1 has to win"""
        game = GameSimulator({'player': False, 'board': (0b011010001, 0b100001110)})
        self.assertEqual(game.play_out(), 1)

    def test_is_even(self):
        """Sets up a almost terminal state that results in no win"""
        game = GameSimulator({'player': True, 'board': (0b011010001, 0b100001110)})
        self.assertEqual(game.play_out(), None)


if __name__ == '__main__':
    unittest.main()
