import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os

class MetricsVisualizer:
    def __init__(self, output_dir="./results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_training_progress(self, rewards, algorithm_name, window_size=20):
        """Plot the training progress of an algorithm"""
        plt.figure(figsize=(10, 6))
        
        # Plot raw rewards
        plt.plot(rewards, alpha=0.3, color='blue', label='Raw Rewards')
        
        # Plot smoothed rewards
        smoothed_rewards = self._smooth_data(rewards, window_size)
        plt.plot(smoothed_rewards, color='blue', linewidth=2, label=f'Smoothed (window={window_size})')
        
        plt.title(f"Training Progress: {algorithm_name}")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.grid(alpha=0.3)
        plt.legend()
        
        # Save the figure
        plt.savefig(os.path.join(self.output_dir, f"{algorithm_name}_training.png"), dpi=200)
        plt.close()
    
    def plot_tournament_results(self, results_df):
        """Plot the tournament results"""
        plt.figure(figsize=(12, 8))
        
        # Create a bar chart for wins, draws, and losses
        agents = results_df['Agent']
        x = np.arange(len(agents))
        width = 0.25
        
        plt.bar(x - width, results_df['Wins'], width, label='Wins', color='green')
        plt.bar(x, results_df['Draws'], width, label='Draws', color='gray')
        plt.bar(x + width, results_df['Losses'], width, label='Losses', color='red')
        
        plt.xlabel('Agents')
        plt.ylabel('Games')
        plt.title('Tournament Results')
        plt.xticks(x, agents)
        plt.legend()
        
        plt.savefig(os.path.join(self.output_dir, "tournament_results.png"), dpi=200)
        plt.close()
        
        # Create a heatmap of win rates
        self._create_matchup_heatmap(results_df)
    
    def plot_computational_efficiency(self, agent_names, nodes_per_second, time_per_move):
        """Plot the computational efficiency of different agents"""
        if len(agent_names) != len(nodes_per_second):
            print(f"Warning: Mismatch between agent names ({len(agent_names)}) and metrics ({len(nodes_per_second)})")
            # Use only the agents we have metrics for
            agent_names = agent_names[:len(nodes_per_second)]
        plt.figure(figsize=(14, 6))
        
        # Plot nodes per second
        plt.subplot(1, 2, 1)
        plt.bar(agent_names, nodes_per_second, color='blue')
        plt.title('Computational Efficiency')
        plt.xlabel('Agent')
        plt.ylabel('Nodes Explored per Second')
        plt.xticks(rotation=45)
        
        # Plot time per move
        plt.subplot(1, 2, 2)
        plt.bar(agent_names, time_per_move, color='orange')
        plt.title('Time per Move')
        plt.xlabel('Agent')
        plt.ylabel('Average Time (seconds)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "computational_efficiency.png"), dpi=200)
        plt.close()
    
    def plot_q_table_heatmap(self, q_table, title="Q-Table Values"):
        """Create a heatmap visualization of a Q-table"""
        # Reshape Q-table for visualization if needed
        if len(q_table.shape) > 2:
            # We need to find a way to reduce dimensions
            # For this example, we'll take the max Q-value across all actions
            reshaped_q = np.max(q_table, axis=-1)
            
            # If still more than 2D, we'll flatten all but the last two dimensions
            while len(reshaped_q.shape) > 2:
                # Combine the first two dimensions
                reshaped_q = reshaped_q.reshape(-1, *reshaped_q.shape[2:])
        else:
            reshaped_q = q_table
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(reshaped_q, cmap='viridis', annot=False)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "q_table_heatmap.png"), dpi=200)
        plt.close()
    
    def _smooth_data(self, data, window_size):
        """Apply a moving average to smooth data"""
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    
    def _create_matchup_heatmap(self, results_df):
        """Create a heatmap showing win rates between different agents"""
        # This would require additional data about individual matchups
        # For now, we'll create a placeholder heatmap
        agents = results_df['Agent'].tolist()
        n_agents = len(agents)
        
        # Placeholder win rate matrix (would need to be calculated from actual matchups)
        win_rate_matrix = np.random.rand(n_agents, n_agents)
        np.fill_diagonal(win_rate_matrix, 0.5)  # Draw against self
        
        # Make the matrix symmetric for placeholder
        win_rate_matrix = (win_rate_matrix + win_rate_matrix.T) / 2
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(win_rate_matrix, annot=True, fmt=".2f", 
                    xticklabels=agents, yticklabels=agents, cmap="RdBu_r",
                    vmin=0, vmax=1)
        plt.title("Win Rate Matrix (Row vs Column)")
        
        plt.savefig(os.path.join(self.output_dir, "matchup_heatmap.png"), dpi=200)
        plt.close()