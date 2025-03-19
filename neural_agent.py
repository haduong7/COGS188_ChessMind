import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import chess
from chess_env import ChessEnvironment

class ChessNN(nn.Module):
    def __init__(self):
        super(ChessNN, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Convolutional backbone
        self.conv1 = nn.Conv2d(14, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Policy head - output a probability distribution over all possible moves
        self.policy_conv = nn.Conv2d(256, 64, kernel_size=1)
        self.policy_fc = nn.Linear(64 * 8 * 8, 1968)  # 1968 = max possible moves
        
        # Value head - evaluate the position
        self.value_conv = nn.Conv2d(256, 64, kernel_size=1)
        self.value_fc1 = nn.Linear(64 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

        self.to(self.device)
    
    def forward(self, x):
        x = x.to(self.device)
        
        # Process input through convolutional layers
        x = x.permute(0, 3, 1, 2)  # Change from (B,8,8,14) to (B,14,8,8)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Policy head
        policy = F.relu(self.policy_conv(x))
        # Change view to reshape
        policy = policy.reshape(-1, 64 * 8 * 8)
        policy = self.policy_fc(policy)
        
        # Value head
        value = F.relu(self.value_conv(x))
        # Change view to reshape
        value = value.reshape(-1, 64 * 8 * 8)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value

class NeuralAgent:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ChessNN().to(self.device)
        
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model.eval()
        self.env = ChessEnvironment()

    def evaluate_batch(self, board_fens):
        """Evaluates multiple board positions at once but also works for single inputs."""
        
        # Check if the input is a single FEN string, convert to a list if necessary
        single_input = False
        if isinstance(board_fens, str): 
            board_fens = [board_fens]
            single_input = True 


        batch_tensors = [torch.FloatTensor(self.env.set_fen(fen).get_state()) for fen in board_fens]
        batch_tensor = torch.stack(batch_tensors).to(self.device)  # Stack into a batch

        # Run a single batch forward pass through the neural network
        with torch.no_grad():
            batch_policies, batch_values = self.model(batch_tensor)  # Single forward pass for all positions

        batch_policies = batch_policies.cpu().numpy()
        batch_values = batch_values.cpu().numpy()

        # If it was a single input, return only the first result
        if single_input:
            return batch_policies[0], batch_values[0]

        return batch_policies, batch_values
    
    def select_move(self, board_fen, temperature=1.0):
        """Select a move using the neural network policy"""
        
        policy, _ = self.evaluate_batch(board_fen)  
        policy = torch.tensor(policy, device="cpu")
        
        # Get legal moves
        action_space = self.env.get_action_space()
        legal_moves = list(action_space.values())
        
        # Filter policy for legal moves only
        legal_policy = torch.zeros(len(action_space), device="cpu")
        for i, move_uci in action_space.items():
            legal_policy[i] = policy[i]
        
        # Apply temperature
        if temperature != 0:
            legal_policy = legal_policy ** (1 / temperature)
        
        # Normalize
        if legal_policy.sum() > 0:
            legal_policy = legal_policy / legal_policy.sum()
        else:
            # If all values are zero, use uniform distribution
            legal_policy = torch.ones(len(action_space)) / len(action_space)
        
        # Sample move or take the highest probability
        if temperature == 0:
            # Deterministic - highest probability move
            move_idx = legal_policy.argmax().item()
        else:
            # Sample based on probabilities
            move_idx = torch.multinomial(legal_policy, 1).item()
        
        return action_space[move_idx]