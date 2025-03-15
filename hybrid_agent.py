import chess
import math
import random
import time
import torch
import numpy as np
from chess_env import ChessEnvironment
from neural_agent import NeuralAgent

class HybridNode:
    def __init__(self, env, neural_agent, parent=None, move=None):
        self.env = env
        self.neural_agent = neural_agent
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = list(env.board.legal_moves)
        
        # Get policy and value from neural network
        if neural_agent:
            try:
                state = env.get_state()
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    self.policy, value = neural_agent.model(state_tensor)
                self.policy = self.policy.squeeze().numpy()
                self.value = value.item()
            except Exception as e:
                print(f"Error getting neural network prediction: {e}")
                self.policy = None
                self.value = None
        else:
            self.policy = None
            self.value = None
    
    def select_child(self, c_param=1.414):
        """Select child using a simpler UCB1 formula"""
        if not self.children:
            return None
        
        # Use a simpler UCB1 formula without the policy prior
        def ucb_score(child):
            exploitation = child.wins / child.visits if child.visits > 0 else 0
            exploration = c_param * math.sqrt(math.log(self.visits) / max(child.visits, 1))
            return exploitation + exploration
        
        try:
            return max(self.children, key=ucb_score)
        except Exception as e:
            # Fallback to most visited child if any error occurs
            print(f"Error in select_child: {e}, falling back to most visited")
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
        child_node = HybridNode(new_env, self.neural_agent, parent=self, move=move)
        self.children.append(child_node)
        return child_node
    
    def update(self, result):
        """Update node statistics"""
        self.visits += 1
        self.wins += result

class HybridAgent:
    def __init__(self, neural_agent, iterations=400):
        self.neural_agent = neural_agent
        self.iterations = iterations
        self.nodes_explored = 0
        self.move_time = 0
        self.name = "Hybrid"  # Add a name attribute for tournament display
    
    def select_move(self, board_fen, time_limit=None):
        """Select the best move using hybrid neural-MCTS approach"""
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
            root = HybridNode(env, self.neural_agent)
        except Exception as e:
            print(f"Error creating root node: {e}")
            # Fallback to random move
            return random.choice(list(env.board.legal_moves)).uci()
        
        # Run MCTS iterations with strict timeout
        iteration_count = 0
        max_time = start_time + time_limit
        
        try:
            while iteration_count < self.iterations and time.time() < max_time:
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
                    
                    # Check for excessively long selection paths
                    if len(search_path) > 100 or time.time() - search_start > time_limit * 0.5:
                        print("Selection taking too long, breaking early")
                        break
                
                # Expansion (with timeout check)
                if node is not None and node.untried_moves and time.time() < max_time:
                    try:
                        node = node.expand()
                        self.nodes_explored += 1
                        if node is not None:
                            search_path.append(node)
                    except Exception as e:
                        print(f"Error in expansion: {e}")
                        break  # Stop this iteration if expansion fails
                
                # Evaluation - use neural net or default to 0.5
                result = 0.5  # Default to draw
                if node is not None:
                    if hasattr(node, 'value') and node.value is not None:
                        result = (node.value + 1) / 2  # Convert from [-1,1] to [0,1]
                    else:
                        # Simple rollout without actual simulation
                        result = 0.5
                
                # Backpropagation (with error handling)
                for n in search_path:
                    if n is not None:  # Safety check
                        try:
                            # Adjust result based on turn
                            adjusted_result = result
                            if n.env.board.turn == chess.BLACK:
                                adjusted_result = 1 - result
                            
                            n.update(adjusted_result)
                        except Exception as e:
                            print(f"Error in backpropagation: {e}")
                
                iteration_count += 1
                
                # Check if we're running out of time
                if time.time() > max_time - 0.1:  # Leave 0.1s margin
                    break
        except Exception as e:
            print(f"Error during MCTS search: {e}")
        
        # Select best move with error handling
        best_move = None
        try:
            if root.children:
                best_child = max(root.children, key=lambda child: child.visits)
                best_move = best_child.move
        except Exception as e:
            print(f"Error selecting best move: {e}")
        
        # Fallback to random move if no best move found
        if best_move is None and list(env.board.legal_moves):
            print("Using random fallback move")
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