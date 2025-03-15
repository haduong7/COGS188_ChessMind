import chess
import math
import random
import time
from chess_env import ChessEnvironment

class MCTSNode:
    def __init__(self, env, parent=None, move=None):
        self.env = env
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = list(env.board.legal_moves)
        random.shuffle(self.untried_moves)
    
    def select_child(self, c_param=1.414):
        """Select a child node using UCB1 formula"""
        try:
            return max(self.children, key=lambda child: 
                      (child.wins / child.visits if child.visits > 0 else 0) + 
                      c_param * math.sqrt(math.log(max(self.visits, 1)) / max(child.visits, 1)))
        except Exception as e:
            # Fallback to most visited child if any error
            if self.children:
                return max(self.children, key=lambda c: c.visits)
            return None
    
    def expand(self):
        """Expand by adding a new child node"""
        if not self.untried_moves:
            return None
            
        move = self.untried_moves.pop()
        new_env = self.env.clone()
        new_env.board.push(move)
        child_node = MCTSNode(new_env, parent=self, move=move)
        self.children.append(child_node)
        return child_node
    
    def update(self, result):
        """Update node statistics"""
        self.visits += 1
        self.wins += result

class MCTSAgent:
    def __init__(self, iterations=1000):
        self.iterations = iterations
        self.nodes_explored = 0
        self.move_time = 0
        self.name = "MCTS"  # Add name for tournament display
    
    def select_move(self, board_fen, time_limit=None):
        """Select the best move using Monte Carlo Tree Search"""
        start_time = time.time()
        self.nodes_explored = 0
        
        # Set default time limit if none provided
        if time_limit is None:
            time_limit = 5.0  # 5 seconds default
        
        # Initialize environment
        env = ChessEnvironment()
        env.set_fen(board_fen)
        
        # If no legal moves, return None
        if not list(env.board.legal_moves):
            return None
        
        # Initialize root node
        try:
            root = MCTSNode(env)
        except Exception as e:
            print(f"Error creating root node: {e}")
            # Fallback to random move
            return random.choice(list(env.board.legal_moves)).uci()
        
        # Run MCTS iterations with timeout
        max_time = start_time + time_limit * 0.95  # Leave 5% margin
        iteration_count = 0
        
        try:
            # Limit iterations to a reasonable number based on time
            max_iterations = min(self.iterations, int(time_limit * 100))
            
            while iteration_count < max_iterations and time.time() < max_time:
                # Selection
                node = root
                search_path = [node]
                
                # Select with timeout check
                search_start = time.time()
                while node.untried_moves == [] and node.children != [] and time.time() < max_time:
                    node = node.select_child()
                    if node is None:
                        break
                    search_path.append(node)
                    
                    # Prevent excessive depth search
                    if len(search_path) > 100 or time.time() - search_start > time_limit * 0.2:
                        break
                
                # Expansion
                if node is not None and node.untried_moves and time.time() < max_time:
                    try:
                        expanded_node = node.expand()
                        if expanded_node:
                            node = expanded_node
                            self.nodes_explored += 1
                            search_path.append(node)
                    except Exception as e:
                        print(f"Error in expansion: {e}")
                        break
                
                # Simulation - quick random rollout
                if node is None:
                    result = 0.5  # Default to draw
                else:
                    try:
                        rollout_depth = 0
                        rollout_env = node.env.clone()
                        max_rollout_depth = 20  # Limit rollout depth for speed
                        
                        while not rollout_env.board.is_game_over() and rollout_depth < max_rollout_depth:
                            legal_moves = list(rollout_env.board.legal_moves)
                            if not legal_moves:
                                break
                            
                            # Quick random move selection
                            random_move = random.choice(legal_moves)
                            rollout_env.board.push(random_move)
                            rollout_depth += 1
                            
                            # Check for timeout
                            if rollout_depth % 5 == 0 and time.time() > max_time:
                                break
                        
                        # Determine result (with simplified scoring)
                        if rollout_env.board.is_checkmate():
                            result = 0 if rollout_env.board.turn == chess.WHITE else 1
                        else:
                            result = 0.5  # Draw or max depth reached
                    except Exception as e:
                        print(f"Error in simulation: {e}")
                        result = 0.5
                
                # Backpropagation
                for n in search_path:
                    if n is not None:
                        try:
                            # Adjust score based on perspective
                            adjusted_result = result
                            if n.env.board.turn == chess.BLACK:
                                adjusted_result = 1 - result
                            
                            n.update(adjusted_result)
                        except Exception as e:
                            print(f"Error in backpropagation: {e}")
                
                iteration_count += 1
                
                # Adaptive early stopping if we have a clear best move
                if iteration_count > 50:
                    best_child = max(root.children, key=lambda c: c.visits) if root.children else None
                    if best_child and best_child.visits > root.visits * 0.8:
                        # One move is getting majority of visits - no need to search more
                        break
        except Exception as e:
            print(f"Error during MCTS search: {e}")
        
        # Select the move with highest visit count
        best_move = None
        try:
            if root.children:
                best_child = max(root.children, key=lambda c: c.visits)
                best_move = best_child.move
        except Exception as e:
            print(f"Error selecting best move: {e}")
        
        # Fallback to random if something went wrong
        if best_move is None and list(env.board.legal_moves):
            best_move = random.choice(list(env.board.legal_moves))
        
        self.move_time = time.time() - start_time
        return best_move.uci() if best_move else None
    
    def get_stats(self):
        """Return statistics about the last move decision"""
        return {
            "nodes_explored": self.nodes_explored,
            "time_taken": self.move_time,
            "nodes_per_second": self.nodes_explored / self.move_time if self.move_time > 0 else 0
        }