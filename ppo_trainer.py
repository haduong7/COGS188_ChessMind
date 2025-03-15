import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import chess
from tqdm import tqdm
from chess_env import ChessEnvironment
from neural_agent import ChessNN

class PPOTrainer:
    def __init__(self, model=None, lr=0.0001, gamma=0.99, clip_ratio=0.2, 
                 value_coef=0.5, entropy_coef=0.01):
        # Initialize model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model if model else ChessNN().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # PPO hyperparameters
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
    
    def collect_experience(self, num_games=50, max_moves=100):
        """Collect experience through self-play"""
        states = []
        actions = []
        rewards = []
        old_log_probs = []
        values = []
        
        for _ in tqdm(range(num_games), desc="Self-play games"):
            env = ChessEnvironment()
            state = env.reset()
            game_states = []
            game_actions = []
            game_log_probs = []
            game_values = []
            
            move_count = 0
            while not env.board.is_game_over() and move_count < max_moves:
                # Convert state for neural network
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                
                # Get action probabilities and value
                with torch.no_grad():
                    policy_logits, value = self.model(state_tensor)
                    policy_logits = policy_logits.squeeze(0)
                    value = value.item()
                
                # Filter for legal moves
                action_space = env.get_action_space()
                legal_policy = torch.zeros(len(policy_logits)).to(self.device)

                for idx, move_uci in action_space.items():
                    legal_policy[idx] = policy_logits[idx]
                
                # Apply softmax to get probabilities
                # Only apply softmax to the legal moves (adding a small epsilon to prevent NaN)
                if torch.sum(legal_policy) > -1e10:  # Check if we have any reasonable policy values
                    legal_probs = F.softmax(legal_policy, dim=0)
                else:
                    # If all legal moves have very bad values, use uniform distribution
                    legal_probs = torch.zeros(len(policy_logits)).to(self.device)
                    for idx in action_space.keys():
                        legal_probs[idx] = 1.0 / len(action_space)
                
                legal_indices = list(action_space.keys())
                if not legal_indices:
                    # No legal moves, game should be over
                    break

                mask = torch.zeros_like(legal_probs)
                for idx in legal_indices:
                    mask[idx] = 1.0

                # Zero out probabilities of illegal moves
                masked_probs = legal_probs * mask

                # Normalize to make sure it sums to 1
                if masked_probs.sum() > 0:
                    masked_probs = masked_probs / masked_probs.sum()
                else:
                    # Fallback to uniform distribution across legal moves
                    for idx in legal_indices:
                        masked_probs[idx] = 1.0 / len(legal_indices)

                # Sample from legal moves only
                action_idx = torch.multinomial(masked_probs, 1).item()
                move_uci = action_space[action_idx]
                
                # Store log probability of selected action
                log_prob = torch.log(legal_probs[action_idx] + 1e-10)
                
                # Take action
                next_state, reward, done = env.step(move_uci)
                
                # Store experience
                game_states.append(state)
                game_actions.append(action_idx)
                game_log_probs.append(log_prob.item())
                game_values.append(value)
                
                # Update state
                state = next_state
                move_count += 1
                
                if done:
                    break
            
            # Calculate game result
            if env.board.is_checkmate():
                final_reward = 1 if env.board.turn == chess.BLACK else -1
            else:
                final_reward = 0  # Draw
            
            # Calculate rewards with intermediate values
            game_rewards = []
            for i in range(len(game_states)):
                # Material advantage reward
                material_score = env.evaluate() / 30  # Scale down material score
                material_score = material_score if i % 2 == 0 else -material_score  # Perspective
                
                # Center control bonus
                # Add more intermediate reward components...
                
                # Final outcome reward (decayed by distance from end)
                outcome_reward = final_reward * (self.gamma ** (len(game_states) - i - 1))
                
                # Combine rewards
                move_reward = 0.2 * material_score + 0.8 * outcome_reward
                
                game_rewards.append(move_reward)
            
            # Append game data to overall experience
            states.extend(game_states)
            actions.extend(game_actions)
            rewards.extend(game_rewards)
            old_log_probs.extend(game_log_probs)
            values.extend(game_values)
        
        return states, actions, rewards, old_log_probs, values
    
    def update_model(self, states, actions, rewards, old_log_probs, values, epochs=4, batch_size=256):
        """Update model using PPO algorithm"""
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(old_log_probs)).to(self.device)
        old_values = torch.FloatTensor(np.array(values)).to(self.device)
        
        # Calculate advantages
        advantages = rewards - old_values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO training loop
        total_loss = 0
        for _ in range(epochs):
            # Generate random indices for mini-batches
            indices = torch.randperm(len(states))
            
            # Process data in mini-batches
            for start_idx in range(0, len(states), batch_size):
                # Get mini-batch
                batch_indices = indices[start_idx:start_idx + batch_size]
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_rewards = rewards[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                
                # Forward pass
                policy_logits, values = self.model(batch_states)
                values = values.squeeze(-1)
                
                # Get new action probabilities
                probs = F.softmax(policy_logits, dim=1)
                batch_log_probs = torch.log(probs.gather(1, batch_actions.unsqueeze(1)).squeeze(1) + 1e-10)
                
                # Entropy for exploration
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1).mean()
                
                # Policy loss with PPO clipping
                ratio = torch.exp(batch_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, batch_rewards)
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
        
        return total_loss / (len(states) // batch_size * epochs)
    
    def train(self, iterations=50, games_per_iteration=50, save_path=None):
        """Train the model using PPO"""
        all_losses = []
        
        for i in range(iterations):
            print(f"Iteration {i+1}/{iterations}")
            
            # Collect experience
            states, actions, rewards, old_log_probs, values = self.collect_experience(num_games=games_per_iteration)
            
            # Update model
            loss = self.update_model(states, actions, rewards, old_log_probs, values)
            all_losses.append(loss)
            
            print(f"Loss: {loss:.4f}")
            
            # Save model
            if save_path and (i+1) % 5 == 0:
                torch.save(self.model.state_dict(), f"{save_path}_iter_{i+1}.pt")
        
        # Save final model
        if save_path:
            torch.save(self.model.state_dict(), f"{save_path}_final.pt")
        
        return all_losses