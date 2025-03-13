import chess
import chess.pgn
import numpy as np
import copy

class ChessEnvironment:
    def __init__(self):
        """Initializes the chess environment with a standard board setup"""
        self.board = chess.Board()
        self.transposition_table = {} 
    
    def reset(self):
        """Resets the board
        
        Returns:
            np.ndarray: The board state as an 8x8x14 tensor
        """
        self.board.reset()
        return self.get_state()
    
    def step(self, move_uci: str):
        """Applies a move in UCI format
        
        Args:
            move_uci (str): Move in UCI format (e.g. "e2e4")
        
        Returns:
            tuple: (np.ndarray, int, bool) representing the new state, reward, and game termination
        """
        if move_uci in [move.uci() for move in self.board.legal_moves]:
            self.board.push_uci(move_uci)
            reward = self.get_reward()
            done = self.board.is_game_over()
            return self.get_state(), reward, done
        else:
            raise ValueError("Invalid move")
    
    
    def get_fen(self):
        """Returns the board state in FEN
        
        Returns:
            str: The board's current FEN string
        """
        return self.board.fen()
    
    def set_fen(self, fen: str):
        """Sets the board state from a FEN string
        
        Args:
            fen (str): A valid FEN string representing a board state.
        """
        self.board.set_fen(fen)
        
        
    def get_reward(self):
        """Returns reward
        
        Returns:
            int: +1 for win, -1 for loss, 0 for draw or ongoing game.
        """
        if self.board.is_checkmate():
            return 1 if self.board.turn == chess.BLACK else -1
        elif self.board.is_stalemate() or self.board.is_insufficient_material():
            return 0
        return 0
    
    
    def get_state(self):
        """NeuralNet PPO: Encodes the board state as an 8x8x14 tensor
        
        Returns:
            np.ndarray: Board state.
        """
        state = np.zeros((8, 8, 14), dtype=np.int8)
        piece_map = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, 
            chess.QUEEN: 4, chess.KING: 5
        }
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                color_offset = 6 if piece.color == chess.BLACK else 0
                row, col = divmod(square, 8)
                state[row, col, piece_map[piece.piece_type] + color_offset] = 1
        return state
    
        
    def evaluate(self):
        """Minimax & MCTS: Evaluates the board heuristically using a transposition table.
        The heuristic is based on material values (pawn=1, knight/bishop=3, rook=5, queen=9) 
        Positive = white, Negative = black
        
        Returns:
            int: The evaluation score of the board state
        """
        board_hash = hash(self.board)
        if board_hash in self.transposition_table:
            return self.transposition_table[board_hash]
        
        piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
        score = 0
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                value = piece_values[piece.piece_type]
                score += value if piece.color == chess.WHITE else -value
        
        self.transposition_table[board_hash] = score
        return score
    
    def order_moves(self):
        """Minimax: Sorts legal moves based on heuristics for better search efficiency.
        Moves are prioritized by Most Valuable Victim - Least Valuable Attacker (MVV-LVA) for captures, 
        with additional priority given to moves that deliver check.
        
        Returns:
            list: A sorted list of legal chess Move objects
        """
        def move_priority(move):
            if self.board.is_capture(move):
                victim = self.board.piece_at(move.to_square)
                attacker = self.board.piece_at(move.from_square)
                if victim and attacker:
                    return 10 * victim.piece_type - attacker.piece_type
            if self.board.gives_check(move):
                return 5
            return 0
        
        moves = list(self.board.legal_moves)
        return sorted(moves, key=move_priority, reverse=True)
    
    

    def clone(self):
        """MCTS: Creates a deep copy of the environment
        
        Returns:
            ChessEnvironment: A new ChessEnvironment instance with the same board state.
        """
        cloned_env = ChessEnvironment()
        cloned_env.set_fen(self.get_fen())
        return cloned_env
    
    
    def get_action_space(self):
        """NeuralNet PPO: Computes the action space for the current board state
        
        Returns:
            dict: A mapping of action indices to UCI move strings
        """
        action_space = {}
        index = 0
        for move in self.board.legal_moves:
            action_space[index] = move.uci()
            index += 1
        return action_space
    
    
    def move_to_action(self, move_uci: str):
        """NeuralNet PPO: Converts a UCI move into an action index
        
        Args:
            move_uci (str): A move string in UCI format.
        
        Returns:
            int or None: The corresponding action index, or None if the move is illegal
        """
        action_space = self.get_action_space()
        for index, move in action_space.items():
            if move == move_uci:
                return index
        return None
    
    
    def action_to_move(self, action_index: int):
        """NeuralNet PPO: Converts an action index back into a UCI move
        
        Args:
            action_index (int): The index of the action.
        
        Returns:
            str or None: The corresponding UCI move, or None if the index is invalid.
        """
        action_space = self.get_action_space()
        return action_space.get(action_index, None)

