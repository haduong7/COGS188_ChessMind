import chess
import math
import random
import time
import torch
import numpy as np
from chess_env import ChessEnvironment
from neural_agent import NeuralAgent

class HybridNode:
    def __init__(self, env, neural_agent, parent=None, move=None, prior=None):
        self.env = env
        self.neural_agent = neural_agent
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = list(env.board.legal_moves)
        
        if prior is not None:
            # Use the provided prior (a scalar) and skip NN evaluation
            self.policy = prior  
            # You might set a default value for the value if desired.
            self.value = 0.5  
        elif neural_agent:
            try:
                state = env.get_state()
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(neural_agent.device)
                with torch.no_grad():
                    policy, value = neural_agent.model(state_tensor)
                # Store the full policy vector for the node
                self.policy = policy.squeeze().cpu().numpy()
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
        
        def puct_score(child):
            """PUCT formula: balances exploration & exploitation"""
            q_value = child.wins / (child.visits + 1)  # Q(s, a)
            prior = child.policy if child.policy is not None else 1 / len(self.children)  # Prior from NN
            ucb_term = c_param * prior * (math.sqrt(self.visits) / (1 + child.visits))
            return q_value + ucb_term

        return max(self.children, key=puct_score)

    def expand(self, batch_nodes=None):
        """Expand all untried moves, using the parent's policy vector to set priors."""
        if batch_nodes is None:
            batch_nodes = []
        
        # Get parent's action space. Assume it returns a dict {index: move_uci}
        action_space = self.env.get_action_space()
        
        # Expand each untried move
        while self.untried_moves:
            move = self.untried_moves.pop()
            # Find the index in the parent's action space that corresponds to this move.
            move_index = None
            for idx, move_uci in action_space.items():
                if move_uci == move.uci():
                    move_index = idx
                    break
            # If not found, use a uniform fallback.
            if move_index is None:
                move_index = 0
            
            # Extract the prior probability for this move from parent's policy vector.
            if self.policy is not None and isinstance(self.policy, np.ndarray):
                prior = float(self.policy[move_index])
            else:
                prior = 1.0 / len(action_space)
            
            new_env = self.env.clone()
            new_env.board.push(move)
            # Create the child node, passing the extracted prior.
            child_node = HybridNode(new_env, self.neural_agent, parent=self, move=move, prior=prior)
            self.children.append(child_node)
            batch_nodes.append(child_node)
        
        return batch_nodes
    
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
                while node.untried_moves == [] and node.children != [] and time.time() < max_time and len(batch_nodes)<16: # batch size 16
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
                        batch_nodes = node.expand([])  
                        self.nodes_explored += len(batch_nodes)  
                
                        # Batch process all expanded nodes
                        if batch_nodes:
                            batch_fens = [n.env.board.fen() for n in batch_nodes]
                            batch_policies, batch_values = self.neural_agent.evaluate_batch(batch_fens)
                            
                            for i, child in enumerate(batch_nodes):
                                p = batch_policies[i]
                                if isinstance(p, np.ndarray):
                                    if p.size == 1:
                                        p = float(p.item())
                                    else:
                                        p = float(np.mean(p)) # take mean value of NN vector of probs
                                else:
                                    p = float(p)
                                child.policy = p
                        
                                v = batch_values[i]
                                if isinstance(v, np.ndarray):
                                    if v.size == 1:
                                        v = float(v.item())
                                    else:
                                        v = float(np.mean(v))
                                else:
                                    v = float(v)
                                child.value = v
                                
                        search_path.extend(batch_nodes) 
                    except Exception as e:
                        print(f"Error in expansion: {e}")
                        break


                result = 0.5  # Default to draw
                
                if node is not None:
                    if hasattr(node, 'value') and node.value is not None:
                        result = (node.value + 1) / 2  # Convert from [-1,1] to [0,1]
                    else:
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