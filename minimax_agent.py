import chess
import time
from chess_env import ChessEnvironment

class MinimaxAgent:
    def __init__(self, max_depth=4):
        self.max_depth = max_depth
        self.nodes_explored = 0
        self.env = ChessEnvironment()
        self.move_time = 0
    
    def select_move(self, board_fen, time_limit=None):
        """Select the best move using minimax with alpha-beta pruning"""
        start_time = time.time()
        self.nodes_explored = 0
        self.env.set_fen(board_fen)
        
        best_move = None
        best_value = float('-inf') if self.env.board.turn == chess.WHITE else float('inf')
        
        # Iterative deepening
        for depth in range(1, self.max_depth + 1):
            if time_limit and time.time() - start_time > time_limit:
                break
                
            value, move = self.iterative_deepening_search(depth)
            
            if self.env.board.turn == chess.WHITE and value > best_value:
                best_value = value
                best_move = move
            elif self.env.board.turn == chess.BLACK and value < best_value:
                best_value = value
                best_move = move
        
        self.move_time = time.time() - start_time
        return best_move.uci() if best_move else None
    
    def iterative_deepening_search(self, depth):
        """Perform minimax search with alpha-beta pruning to a specific depth"""
        is_maximizing = self.env.board.turn == chess.WHITE
        if is_maximizing:
            return self.alpha_beta_max(depth, float('-inf'), float('inf')), self.best_move
        else:
            return self.alpha_beta_min(depth, float('-inf'), float('inf')), self.best_move
    
    def alpha_beta_max(self, depth, alpha, beta):
        """Alpha-beta pruning for maximizing player (White)"""
        self.nodes_explored += 1
        
        # Check for terminal states
        if self.env.board.is_game_over():
            if self.env.board.is_checkmate():
                return -10000  # White is checkmated
            return 0  # Draw
        
        if depth == 0:
            return self.env.evaluate()
        
        max_value = float('-inf')
        self.best_move = None
        
        # Sort moves for better pruning efficiency
        for move in self.env.order_moves():
            self.env.board.push(move)
            value = self.alpha_beta_min(depth - 1, alpha, beta)
            self.env.board.pop()
            
            if value > max_value:
                max_value = value
                if depth == self.max_depth:
                    self.best_move = move
            
            alpha = max(alpha, value)
            if beta <= alpha:
                break
        
        return max_value
    
    def alpha_beta_min(self, depth, alpha, beta):
        """Alpha-beta pruning for minimizing player (Black)"""
        self.nodes_explored += 1
        
        # Check for terminal states
        if self.env.board.is_game_over():
            if self.env.board.is_checkmate():
                return 10000  # Black is checkmated
            return 0  # Draw
        
        if depth == 0:
            return self.env.evaluate()
        
        min_value = float('inf')
        
        # Sort moves for better pruning efficiency
        for move in self.env.order_moves():
            self.env.board.push(move)
            value = self.alpha_beta_max(depth - 1, alpha, beta)
            self.env.board.pop()
            
            if value < min_value:
                min_value = value
                if depth == self.max_depth:
                    self.best_move = move
            
            beta = min(beta, value)
            if beta <= alpha:
                break
        
        return min_value
    
    def get_stats(self):
        """Return statistics about the last move decision"""
        return {
            "nodes_explored": self.nodes_explored,
            "time_taken": self.move_time,
            "nodes_per_second": self.nodes_explored / self.move_time if self.move_time > 0 else 0
        }