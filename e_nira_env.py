import numpy as np
import random

# Define valid movement pairs
VALID_MOVES = {
    1: [2, 4, 5], 2: [1, 3, 5], 3: [2, 5, 6],
    4: [1, 5, 7], 5: [1, 2, 3, 4, 6, 7, 8, 9], 6: [3, 5, 9],
    7: [4, 5, 8], 8: [5, 7, 9], 9: [5, 6, 8]
}

MOVE_ACTIONS = [(i, j) for i in range(9) for j in range(9)
                if (j + 1) in VALID_MOVES[i + 1] and i != j]
MOVE_ACTION_INDEX = {pair: idx for idx, pair in enumerate(MOVE_ACTIONS)}
INDEX_TO_MOVE_ACTION = {v: k for k, v in MOVE_ACTION_INDEX.items()}

class ENiraEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros(9, dtype=int)
        self.turn = 1  # 1 = human, -1 = AI
        self.phase = 'placing' # 'placing' or 'moving'
        self.placed_count = {1: 0, -1: 0} # Tracks how many pieces each player has placed
        self.steps = 0
        return self._get_state()

    def _get_state(self):
        # State includes: board (9 values), current turn (1 value), current phase (2 values via one-hot/binary)
        # Using 0 for placing, 1 for moving for simplicity.
        phase_numeric = 0 if self.phase == 'placing' else 1
        # Concatenate board state with turn and phase
        return np.append(self.board, [self.turn, phase_numeric])

    def _check_win(self, player):
        """Checks if the given player has formed a winning line (mill)."""
        wins = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # columns
            [0, 4, 8], [2, 4, 6]              # diagonals
        ]
        # Check if the player occupies any of the winning combinations
        for w in wins:
            if all(self.board[p] == player for p in w):
                return True
        return False


    def _check_mill(self, board_state, player):
        """Counts the number of mills a player has on a given board state."""
        wins = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # columns
            [0, 4, 8], [2, 4, 6]              # diagonals
        ]
        count = 0
        for w in wins:
            if all(board_state[p] == player for p in w):
                count += 1
        return count

    def _check_potential_mill(self, board_state, player):
        """Checks for two pieces in a line with the third spot empty."""
        wins = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # columns
            [0, 4, 8], [2, 4, 6]              # diagonals
        ]
        potential_mills = 0
        for w in wins:
            player_pieces_in_line = [pos for pos in w if board_state[pos] == player]
            empty_spot_in_line = [pos for pos in w if board_state[pos] == 0]
            
            if len(player_pieces_in_line) == 2 and len(empty_spot_in_line) == 1:
                # Make sure the empty spot is connected to the two player pieces in the mill
                # This might need more sophisticated graph traversal or pre-defined valid connections
                # For a simple check, we assume the 'wins' list implicitly means they are connected.
                potential_mills += 1
        return potential_mills

    def get_valid_actions(self):
        if self.phase == "placing":
            return [i for i in range(9) if self.board[i] == 0]
        else: # moving phase
            actions = []
            for (from_idx, to_idx) in MOVE_ACTIONS:
                if self.board[from_idx] == self.turn and self.board[to_idx] == 0:
                    actions.append(9 + MOVE_ACTION_INDEX[(from_idx, to_idx)])
            return actions

    def step(self, action):
        done = False
        reward = 0
        
        current_player = self.turn # Store current player for reward calculations before turn switch

        previous_board = self.board.copy()
        previous_mills_current_player = self._check_mill(previous_board, current_player)
        previous_mills_opponent = self._check_mill(previous_board, -current_player)
        previous_potential_mills_current_player = self._check_potential_mill(previous_board, current_player)
        previous_potential_mills_opponent = self._check_potential_mill(previous_board, -current_player)


        # --- Execute the action ---
        if self.phase == "placing":
            if self.board[action] != 0: # Invalid action: trying to place on an occupied spot
                return self._get_state(), -10, True, {} # Increased penalty for invalid moves
            self.board[action] = current_player
            self.placed_count[current_player] += 1
        else: # moving phase
            move_idx = action - 9
            if move_idx < 0 or move_idx >= len(MOVE_ACTIONS): # Invalid action index
                return self._get_state(), -10, True, {}
            from_idx, to_idx = INDEX_TO_MOVE_ACTION[move_idx]
            if self.board[from_idx] != current_player or self.board[to_idx] != 0: # Invalid move
                return self._get_state(), -10, True, {}

            self.board[from_idx] = 0
            self.board[to_idx] = current_player

        # --- Check for immediate win/loss ---
        if self._check_win(current_player): # THIS IS THE LINE THAT WAS CAUSING THE ERROR
            return self._get_state(), 10, True, {} # Big reward for winning

        # --- Reward Shaping (for current_player's perspective) ---
        current_mills_current_player = self._check_mill(self.board, current_player)
        current_mills_opponent = self._check_mill(self.board, -current_player)
        current_potential_mills_current_player = self._check_potential_mill(self.board, current_player)
        current_potential_mills_opponent = self._check_potential_mill(self.board, -current_player)

        # Reward for forming a new mill
        if current_mills_current_player > previous_mills_current_player:
            reward += 1.0 # Significant reward for completing a mill

        # Penalty for opponent forming a new mill (AI wants to prevent this)
        if current_mills_opponent > previous_mills_opponent:
            reward -= 1.0 # Significant penalty if opponent forms a mill

        # Reward for creating a new potential mill (2 pieces in a line, 1 empty spot)
        if current_potential_mills_current_player > previous_potential_mills_current_player:
            reward += 0.2 # Small reward for setting up
        
        # Penalty for allowing opponent to create a new potential mill
        if current_potential_mills_opponent > previous_potential_mills_opponent:
            reward -= 0.2

        # Small penalty for each step to encourage faster wins
        reward -= 0.01

        # --- Phase Transition ---
        if self.phase == "placing" and self.placed_count[1] == 3 and self.placed_count[-1] == 3:
            self.phase = "moving"

        # --- Switch Turns ---
        self.turn *= -1
        self.steps += 1

        # --- Check for game end conditions ---
        if self.steps >= 200: # Max steps reached
            return self._get_state(), -5, True, {} # Penalty for draw/time out

        # Check if current player (who just made the move) has created a state where the *next* player has no valid moves
        # This is a strategic win, or a form of 'stalemate' where the opponent cannot move.
        # This rule might vary by specific game variant. In basic E-Nira, as long as pieces can move, it's not a win.
        # However, if it leads to inability to move any pieces:
        # if not self.get_valid_actions() and self.turn == -1: # If it's now AI's turn and AI has no moves (human wins)
        #     return self._get_state(), -10, True, {} # Strong penalty if AI is stuck
        # elif not self.get_valid_actions() and self.turn == 1: # If it's now human's turn and human has no moves (AI wins)
        #     return self._get_state(), 10, True, {} # Strong reward if AI blocks human

        return self._get_state(), reward, False, {}