import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import chess
from tqdm import tqdm
from chess_env import ChessEnvironment
from neural_agent import ChessNN
import os
import pandas as pd
import random
#python main.py --mode train --iterations 600 --games_per_iteration 100 --learning_rate 0.0001 --gamma 0.99 --evaluation_interval 30 
class PPOTrainer:
    def __init__(self, model=None, lr=0.0001, gamma=0.99, clip_ratio=0.2, 
                 value_coef=0.5, entropy_coef=0.01, evaluation_interval=10):
        # Initialize model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model if model else ChessNN().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # PPO hyperparameters
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        # Elo tracking
        self.elo_ratings = {
            "current_model": 1500,  # Initial Elo
            "best_model": 1500
        }

        self.evaluation_interval = evaluation_interval # iters to evaluate
    
    def collect_experience(self, num_games=50, max_moves=100):
        """Collect experience using batch processing for PPO"""
    
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_log_probs = []
        batch_values = []
        
        envs = [ChessEnvironment() for _ in range(num_games)]  # Parallel environments
        states = [env.reset() for env in envs]  # Reset all environments
    
        move_counts = [0] * num_games  # Track moves per game
        active_games = list(range(num_games))  # Games still in progress
    
        while active_games:
            states_np = np.array(states, dtype=np.float32)
            state_tensors = torch.tensor(states_np, dtype=torch.float32, device=self.device)  # Batch state input
    
            with torch.no_grad():
                policy_logits, values = self.model(state_tensors)
            
            policy_logits = policy_logits.cpu().numpy()
            values = values.cpu().numpy()
    
            new_active_games = []
            for i, env_index in enumerate(active_games):
                env = envs[env_index]
                move_count = move_counts[env_index]
    
                # Ensure legal moves
                legal_moves = list(env.board.legal_moves)
                if not legal_moves:
                    continue  # Skip games that are finished
                
                # Extract valid action indices
                action_space = env.get_action_space()
                legal_indices = list(action_space.keys())
                legal_probs = np.exp(policy_logits[i][legal_indices])
                legal_probs /= np.sum(legal_probs)  # Normalize
                
                # Sample action
                action_idx = np.random.choice(legal_indices, p=legal_probs)
                move_uci = action_space[action_idx]
                
                # Log probability of action
                log_prob = np.log(legal_probs[action_idx] + 1e-10)
    
                # Take action
                next_state, reward, done = env.step(move_uci)
                
                # Store experience
                batch_states.append(states[env_index])
                batch_actions.append(action_idx)
                batch_rewards.append(reward)
                batch_log_probs.append(log_prob)
                batch_values.append(values[i])
    
                states[env_index] = next_state  # Update state
                move_counts[env_index] += 1
    
                if not done and move_counts[env_index] < max_moves:
                    new_active_games.append(env_index)  # Keep game active
    
            active_games = new_active_games  # Update active games list
    
        return batch_states, batch_actions, batch_rewards, batch_log_probs, batch_values
        
        
    def update_model(self, states, actions, rewards, old_log_probs, values, epochs=4, batch_size=256):
        """Update model using PPO with batch processing and GPU optimization."""
    
        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.long, device=self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device)
        old_log_probs = torch.tensor(np.array(old_log_probs), dtype=torch.float32, device=self.device)
        old_values = torch.tensor(np.array(values), dtype=torch.float32, device=self.device)
    
        advantages = rewards - old_values
        advantages = advantages.view(-1)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
        scaler = torch.cuda.amp.GradScaler()
    
        total_loss = 0
        indices = torch.randperm(len(states))  # Shuffle data
        num_batches = len(states) // batch_size  # Ensure consistent batch size
    
        for _ in tqdm(range(epochs), desc="Training Epochs"):
            for i in tqdm(range(num_batches), desc="Processing Mini-batches", leave=False):
                batch_indices = indices[i * batch_size: (i + 1) * batch_size]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices].view(batch_size)  # FIXED DIMENSION
                batch_rewards = rewards[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
    
                with torch.cuda.amp.autocast():
                    policy_logits, values = self.model(batch_states)
                    values = values.squeeze(-1)
    
                    probs = F.softmax(policy_logits, dim=1)
                    batch_log_probs = torch.log(probs.gather(1, batch_actions.unsqueeze(1)).squeeze(1) + 1e-10)
    
                    ratio = torch.exp(batch_log_probs - batch_old_log_probs)

                    surr1 = ratio * batch_advantages  # FIXED ERROR
                    surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
    
                    value_loss = F.mse_loss(values, batch_rewards)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1).mean()
    
                    loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
    
                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
    
                total_loss += loss.item()
    
        return total_loss / (num_batches * epochs)


    def evaluate_elo(self, iteration, save_path, num_games=30, max_moves=200):
        """Evaluates the current PPO model against the best model and updates Elo ratings."""
    
        best_model_path = f"{save_path}_best.pt"
    
        # If no best model exists, initialize it with the current model
        if not os.path.exists(best_model_path):
            torch.save(self.model.state_dict(), best_model_path)
            self.elo_ratings["best_model"] = self.elo_ratings["current_model"]
            return
    
        # Load best model
        best_model = ChessNN().to(self.device)
        best_model.load_state_dict(torch.load(best_model_path))
    
        current_wins = 0
        best_wins = 0
        draws = 0
    
        # Play games alternating colors
        for game_idx in range(num_games):
            env = ChessEnvironment()
            state = env.reset()
    
            # Assign colors: current model plays White first, then Black
            current_model_white = game_idx % 2 == 0
            move_count = 0
    
            while not env.board.is_game_over() and move_count < max_moves:
                # Choose the model for this move
                current_turn_model = self.model if (env.board.turn == chess.WHITE) == current_model_white else best_model
    
                # Convert state for neural network
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
    
                # Get action probabilities from the model
                with torch.no_grad():
                    policy_logits, _ = current_turn_model(state_tensor)
                    policy_logits = policy_logits.squeeze(0)
    
                # Filter legal moves
                action_space = env.get_action_space()
                legal_policy = torch.zeros(len(policy_logits)).to(self.device)
                for idx, move_uci in action_space.items():
                    legal_policy[idx] = policy_logits[idx]
    
                # Apply softmax to get probabilities
                if torch.sum(legal_policy) > -1e10:
                    legal_probs = F.softmax(legal_policy, dim=0)
                else:
                    legal_probs = torch.zeros(len(policy_logits)).to(self.device)
                    for idx in action_space.keys():
                        legal_probs[idx] = 1.0 / len(action_space)
    
                legal_indices = list(action_space.keys())
                
                if legal_indices:  # Ensure there are legal moves
                    action_idx = torch.multinomial(legal_probs[legal_indices], 1).item()
                    action_idx = legal_indices[action_idx]  # Map back to the valid action space
                    move_uci = action_space[action_idx]
                else:
                    # No legal moves available, game must be over
                    break
    
                # Play the move
                state, _, done = env.step(move_uci)
    
                move_count += 1
                if done:
                    break
    
            # Determine the winner
            if env.board.is_checkmate():
                winner = "current" if env.board.turn == chess.BLACK else "best"  # The opponent won
            else:
                winner = "draw"
    
            if winner == "current":
                current_wins += 1
            elif winner == "best":
                best_wins += 1
            else:
                draws += 1
    
        # Update Elo ratings
        if current_wins > best_wins and current_wins >= num_games * 0.55:  # model has to win 55+% of games
            new_elo, best_elo = self.calculate_elo(
                self.elo_ratings["current_model"], self.elo_ratings["best_model"]
            )
            self.elo_ratings["current_model"], self.elo_ratings["best_model"] = new_elo, best_elo
            torch.save(self.model.state_dict(), best_model_path)
    
        elif best_wins > current_wins:  # If best model is still stronger
            new_best_elo, new_current_elo = self.calculate_elo(
                self.elo_ratings["best_model"], self.elo_ratings["current_model"]
            )
            self.elo_ratings["best_model"], self.elo_ratings["current_model"] = new_best_elo, new_current_elo

        # saves elo to a csv
        pd.DataFrame([{"iteration": iteration, "best_elo": self.elo_ratings["best_model"]}]).to_csv(
            "models/best_elo.csv", index=False, mode="a", header=not os.path.exists("models/best_elo.csv")
        )
        
        print("eval", self.elo_ratings['best_model'])


    def calculate_elo(self, winner_elo, loser_elo, K=32, draw=False):
        """Updates Elo ratings after a match."""
        expected_winner = 1 / (1 + 10 ** ((loser_elo - winner_elo) / 400))
        expected_loser = 1 - expected_winner  # Expected score for the loser
    
        if draw:
            winner_score, loser_score = 0.5, 0.5
        else:
            winner_score, loser_score = 1, 0
    
        new_winner_elo = winner_elo + K * (winner_score - expected_winner)
        new_loser_elo = loser_elo + K * (loser_score - expected_loser)
    
        return round(new_winner_elo, 2), round(new_loser_elo, 2)
        
    
    def train(self, iterations=50, games_per_iteration=50, save_path=None):
        """Train the model using PPO"""
        all_losses = []
        start_iteration = 0
        best_model_path = f"{save_path}_best.pt" if save_path else None 

        if save_path and os.path.exists(f"{save_path}_latest.pt"):
            checkpoint = torch.load(f"{save_path}_latest.pt")
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.elo_ratings["best_model"] = checkpoint.get('best_elo', 1500)
            start_iteration = checkpoint['iteration']
            print(f"Resuming training from iteration {start_iteration+1}")
            
        for i in range(start_iteration, iterations):
            print(f"Iteration {i+1}/{iterations}")
            
            # Collect experience
            states, actions, rewards, old_log_probs, values = self.collect_experience(num_games=games_per_iteration)
            
            # Update model
            loss = self.update_model(states, actions, rewards, old_log_probs, values)
            all_losses.append(loss)
            
            print(f"Loss: {loss:.4f}")
            
            # Save model (save latest and periodically)
            if save_path and (i+1) % 50 == 0:
                torch.save(self.model.state_dict(), f"{save_path}_iter_{i+1}.pt")
                
                torch.save({
                    'iteration': i+1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_elo': self.elo_ratings["best_model"]
                }, f"{save_path}_latest.pt")

            # Elo evaluation every `evaluation_interval` iterations
            if (i + 1) % self.evaluation_interval == 0 and best_model_path:
                self.evaluate_elo((i+1), save_path)
        
        # Save final model
        if save_path:
            torch.save(self.model.state_dict(), f"{save_path}_final.pt")
        
        return all_losses