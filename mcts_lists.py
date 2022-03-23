"""Implementation of Monte Carlo Tree Seach of Tic Tac Toe"""
from __future__ import annotations
# import sys
import math
import random
from time import perf_counter


CONVERT_MOVE = {
    '0 0': 0b100000000,
    '1 0': 0b010000000,
    '2 0': 0b001000000,
    '0 1': 0b000100000,
    '1 1': 0b000010000,
    '2 1': 0b000001000,
    '0 2': 0b000000100,
    '1 2': 0b000000010,
    '2 2': 0b000000001,
    '-1 -1': None,
    0b100000000: '0 0',
    0b010000000: '1 0',
    0b001000000: '2 0',
    0b000100000: '0 1',
    0b000010000: '1 1',
    0b000001000: '2 1',
    0b000000100: '0 2',
    0b000000010: '1 2',
    0b000000001: '2 2',
}

WIN_STATE = (
    0b111000000,
    0b000111000,
    0b000000111,
    0b100100100,
    0b010010010,
    0b001001001,
    0b001010100,
    0b100010001,
)

WIN_STATE_MOVE = {
    0b100000000: (0b111000000, 0b100100100, 0b100010001),
    0b010000000: (0b111000000, 0b010010010),
    0b001000000: (0b111000000, 0b001001001, 0b001010100),
    0b000100000: (0b000111000, 0b100100100),
    0b000010000: (0b000111000, 0b010010010, 0b100010001, 0b001010100),
    0b000001000: (0b000111000, 0b001001001),
    0b000000100: (0b000000111, 0b100100100, 0b001010100),
    0b000000010: (0b000000111, 0b010010010),
    0b000000001: (0b000000111, 0b001001001, 0b100010001)
}

POSSIBLE_MOVES = (
    0b100000000,
    0b010000000,
    0b001000000,
    0b000100000,
    0b000010000,
    0b000001000,
    0b000000100,
    0b000000010,
    0b000000001,
)


class GameState:
    """
    Tic Tac Toe gamestate, the moves are recorded as two 9 bit values.
    Where 0b100000000 is equal to position '0 0' (top left corner), and
    0b000000001 is equal to position '2 2' (bottom right corener).
    The player who put down the move in the state is noted with a boolean.
    """

    def __init__(self, state=None) -> None:
        if state is None:
            self.player = False
            self.board = (0b000000000, 0b000000000)
        else:
            self.player = state['player']
            self.board = state['board']

    def __repr__(self) -> str:
        return f'Player: {self.player} State: {bin(self.board[0])}, {bin(self.board[1])}'

    def get_player(self) -> bool:
        """Gets the current player, for the state"""
        return self.player

    def get_state(self):
        """Gets the state as a dictionary"""
        return {'player': self.player, 'board': self.board}

    def _check_win_move(self, bit_location: int) -> bool:
        """Checks if the last move won the game, returns True if won"""
        won = False
        for state in WIN_STATE_MOVE[bit_location]:
            won |= (self.board[self.player] & state) == state
        return won

    def is_won(self) -> bool:
        """Checks if there is a win in the current gamestate, returns True if won"""
        for state in WIN_STATE:
            player_0_won = (self.board[0] & state) == state
            player_1_won = (self.board[1] & state) == state
            if player_0_won or player_1_won:
                return True
        return False

    def _get_available(self) -> int:
        """Finds available bits where none of the players has put a mark."""
        return (self.board[0] | self.board[1]) ^ 0b111111111

    def get_available_moves(self) -> list:
        """Takes the available bits, and converts it to a list of possible moves"""
        available_bits = self._get_available()
        available_moves = []
        for move in POSSIBLE_MOVES:
            if move & available_bits == move:
                available_moves.append(move)
        return available_moves

    def is_terminated(self) -> bool:
        """Checks if the game is won or that there is a draw"""
        return self.is_won() or self._get_available() == 0

    def move(self, bit_location: int) -> dict:
        """Takes the current gamestate and returns a new state with the move applied"""
        if self.player:
            new_board = (self.board[not self.player] |
                         bit_location, self.board[self.player])
        else:
            new_board = (self.board[self.player],
                         self.board[not self.player] | bit_location)
        return {'player': not self.player, 'board': new_board}


class GameSimulator(GameState):
    """Gamestate that allows simulation to be run on current board"""

    def __init__(self, state=None) -> None:
        super().__init__(state)
        self.board = [0b000000000, 0b000000000]
        self.player = state['player']
        self.board[0] = state['board'][0]
        self.board[1] = state['board'][1]

    def __repr__(self) -> str:
        return f'Player: {self.player} State: {bin(self.board[0])}, {bin(self.board[1])}'

    def move(self, bit_location: int) -> bool:
        """Alternative move function where the input move is applied to the gamestate"""
        self.board[not self.player] |= bit_location
        self.player = not self.player
        return self.is_terminated()

    def _simulate_move(self) -> bool:
        """Plays a random move out of available positions"""
        move = random.choice(self.get_available_moves())
        return self.move(move)

    def play_out(self) -> int:
        """Simulates current board till terminal state"""
        terminated = False
        while not terminated:
            terminated = self._simulate_move()
        if self.is_won():
            result = self.get_player()
        else:
            result = None
        return result


class Node:
    """Monte Carlo Tree search node"""

    def __init__(self, gamestate=None, parent=None, played_move=None) -> None:
        if gamestate is None:
            self.gamestate = GameState()
        else:
            self.gamestate = gamestate
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0
        self.possible_moves = self.gamestate.get_available_moves()
        if self.possible_moves == [] or self.gamestate.is_won():
            self.is_leaf = True
        else:
            self.is_leaf = False
        self.played_move = played_move
        self.uct = 0

    def __repr__(self) -> str:
        return f'{self.gamestate} {CONVERT_MOVE[self.played_move]}'

    def __lt__(self, other) -> bool:
        return self.uct < other.uct

    def __eq__(self, other) -> bool:
        return self.uct == other.uct

    def select(self, constant) -> None:
        """Finds the best (according to uct) child node recursively untill a leaf is reached"""
        if self.children and not self.possible_moves:
            for child_node in self.children:
                child_node.calc_uct(constant)
            max(self.children).select(constant)
        else:
            self.expand()

    def expand(self) -> None:
        """
        Takes a node with possible moves, and expands one randomly to a child node.
        If current node is a leaf, it backpropagates the results up the tree.
        """
        if not self.is_leaf:
            random_move = self.possible_moves.pop(
                random.choice(range(len(self.possible_moves))))
            new_gamestate = GameState(self.gamestate.move(random_move))
            self.children.append(
                Node(gamestate=new_gamestate, parent=self, played_move=random_move))
            self.children[-1].simulate()
        else:
            if self.gamestate.is_won():
                result = self.gamestate.get_player()
            else:
                result = None
            self.backprop(result)

    def simulate(self) -> None:
        """
        Takes current gamestate and simulates till terminal position,
        then backpropagates results.
        """
        simulation_game = GameSimulator(self.gamestate.get_state())
        if not simulation_game.is_terminated():
            result = simulation_game.play_out()
        else:
            if simulation_game.is_won():
                result = not simulation_game.get_player()
            else:
                result = None

        self.backprop(result)

    def backprop(self, result: int) -> None:
        """Uses results from simulation and backpropogates these up the tree."""
        self.visits += 1
        if result == self.gamestate.get_player():
            self.wins += 1
        if self.parent:
            self.parent.backprop(result)

    def calc_uct(self, constant) -> None:
        """calculates UCT value from a node."""
        if self.visits == 0:
            self.visits = 0.001
        self.uct = (self.wins/self.visits) + constant * \
            math.sqrt(math.log(self.parent.visits)/self.visits)


class MonteCarloTreeSearch:
    """
    Basic implementation of Monte Carlo Tree Search
    """
    def __init__(self, exploration_constant=0.8, calculation_time=.09) -> None:
        self.root = Node()
        self.constant = exploration_constant
        self.calc_time = calculation_time
        self.first_move = True
        self.iterations = 0

    def swap_root(self, next_node) -> None:
        """Takes a new node, and swaps the root to the next node."""
        self.iterations = self.root.visits
        self.root = next_node
        self.root.parent = None

    def set_root_to_move(self, move: int) -> None:
        """
        Takes an input move, and swaps the root Node to the node with this move
        if it exists in the child nodes. Otherwise it makes a new node with the move,
        and sets it to root.
        """
        next_node = None
        for child in self.root.children:
            if child.played_move == move:
                next_node = child
                break
        if next_node is None:
            next_state = self.root.gamestate.move(move)
            next_node = Node(gamestate=GameState(
                next_state), played_move=move)
        self.swap_root(next_node)

    def choose_best(self) -> int:
        """Chooses best move from root node, returns the move."""
        for child in self.root.children:
            child.calc_uct(0)
        self.swap_root(max(self.root.children))
        return self.root.played_move

    def run(self) -> int:
        """
        While time limit has not been reached runs through iterations of roll out.
        Then returns best found move.
        """
        start = perf_counter()
        while perf_counter() - start < self.calc_time + .9 * self.first_move:
            self.root.select(self.constant)
        self.first_move = False
        return self.choose_best()


class PlayGame:
    """Deals with everything for actually playing the game"""
    def __init__(self) -> None:
        self.mcts = MonteCarloTreeSearch()

    def move(self, input_move):
        """
        Takes an input move, converts it to binary and swaps the root
        to the correct node. Then runs monte carlo tree search and
        returns a counter move in the correct format.
        """
        converted_move = CONVERT_MOVE[input_move]
        if converted_move:
            self.mcts.set_root_to_move(converted_move)
        return CONVERT_MOVE[self.mcts.run()]

    def play_user(self) -> None:
        """
        Takes user input and returns moves untill the game is terminated.
        Prints all the moves and extra info to the command line.
        """
        while True:
            user_input = input('Enter Move: ')
            print(f'Computer plays: {self.move(user_input)}')
            print(f'Iterations: {self.mcts.iterations}')
            if self.mcts.root.gamestate.is_won():
                print(
                    f'Game is won by {self.mcts.root.gamestate.get_player()}')
                break

    def play_online(self) -> None:
        """
        Version of the play function aimed at use with codinggame.
        Takes all the input from the console and prints just the
        counter move to the console.
        """
        # To debug: print("Debug messages...", file=sys.stderr, flush=True)
        while True:
            # opponent_row, opponent_col = [int(i) for i in input().split()]
            print(self.move(input()))
            valid_action_count = int(input())
            for _ in range(valid_action_count):
                input()
                # row, col = [int(j) for j in input().split()]


if __name__ == '__main__':
    game = PlayGame()
    game.play_user()
