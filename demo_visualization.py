"""
demo_visualization.py - Examples of how to use the visualization tools

Run this script to see demonstrations of the visualization capabilities:
python demo_visualization.py
"""

import os
import chess
import numpy as np
import matplotlib.pyplot as plt
from minimax_agent import MinimaxAgent
from mcts_agent import MCTSAgent
from neural_agent import NeuralAgent, ChessNN
from hybrid_agent import HybridAgent
from visualization import display_board, visualize_game, create_game_gif
from metrics_visualizer import MetricsVisualizer
from game_recorder import GameRecorder

def demo_board_display():
    """Demonstrate the basic board display function"""
    print("Demonstrating basic board display...")
    board = chess.Board()
    
    # Make some moves
    moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6"]
    last_move = None
    
    for uci in moves:
        move = chess.Move.from_uci(uci)
        board.push(move)
        last_move = move
    
    # Display the board
    display_board(board, title="Board after 3 moves each", last_move=last_move)
    input("Press Enter to continue...")

def demo_game_visualization(delay=0.5):
    """Demonstrate the game visualization between two agents"""
    print("Demonstrating game visualization between Minimax and MCTS agents...")
    
    # Initialize agents
    minimax_agent = MinimaxAgent(max_depth=3)
    minimax_agent.name = "Minimax"
    
    mcts_agent = MCTSAgent(iterations=100)
    mcts_agent.name = "MCTS"
    
    # Visualize a game
    board, moves = visualize_game(minimax_agent, mcts_agent, max_moves=10, delay=delay)
    input("Press Enter to continue...")

def demo_game_gif():
    """Demonstrate creating a GIF of a game"""
    print("Creating a game GIF (this may take a moment)...")
    
    # Initialize agents
    minimax_agent = MinimaxAgent(max_depth=2)
    minimax_agent.name = "Minimax"
    
    mcts_agent = MCTSAgent(iterations=50)
    mcts_agent.name = "MCTS"
    
    # Create a GIF
    output_dir = "./demo_results"
    os.makedirs(output_dir, exist_ok=True)
    gif_path = create_game_gif(minimax_agent, mcts_agent, 
                              filename=os.path.join(output_dir, "demo_game.gif"),
                              max_moves=10,
                              frame_duration=1000)
    
    print(f"GIF created at: {gif_path}")
    print("Open this file to view the animated game")
    input("Press Enter to continue...")

def demo_metrics_visualization():
    """Demonstrate metrics visualization"""
    print("Demonstrating metrics visualization...")
    
    # Create a metrics visualizer
    output_dir = "./demo_results"
    os.makedirs(output_dir, exist_ok=True)
    metrics = MetricsVisualizer(output_dir)
    
    # Generate some fake training progress data
    np.random.seed(42)
    episodes = 100
    rewards = np.random.normal(loc=np.linspace(-10, 90, episodes), scale=20)
    
    # Plot training progress
    metrics.plot_training_progress(rewards, "Demo Agent")
    print(f"Training progress plot saved to {output_dir}/Demo_Agent_training.png")
    
    # Create fake tournament results
    import pandas as pd
    results_df = pd.DataFrame({
        'Agent': ['Minimax', 'MCTS', 'Neural', 'Hybrid'],
        'Wins': [15, 22, 18, 25],
        'Draws': [8, 10, 12, 9],
        'Losses': [17, 8, 10, 6],
        'Points': [19, 27, 24, 29.5],
        'Win Rate': ['37.5%', '55.0%', '45.0%', '62.5%']
    })
    
    # Plot tournament results
    metrics.plot_tournament_results(results_df)
    print(f"Tournament results plot saved to {output_dir}/tournament_results.png")
    
    # Plot computational efficiency
    agent_names = ['Minimax', 'MCTS', 'Neural', 'Hybrid']
    nodes_per_second = [5000, 2000, 8000, 3500]
    time_per_move = [0.2, 0.5, 0.1, 0.3]
    
    metrics.plot_computational_efficiency(agent_names, nodes_per_second, time_per_move)
    print(f"Computational efficiency plot saved to {output_dir}/computational_efficiency.png")
    
    input("Press Enter to continue...")

def demo_game_recording():
    """Demonstrate game recording"""
    print("Demonstrating game recording...")
    
    # Initialize recorder
    output_dir = "./demo_results/games"
    os.makedirs(output_dir, exist_ok=True)
    recorder = GameRecorder(output_dir)
    
    # Create a sample game
    board = chess.Board()
    moves = []
    
    # Play some moves
    uci_moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "d2d4", "e5d4", "e1g1", "f6e4"]
    for uci in uci_moves:
        move = chess.Move.from_uci(uci)
        board.push(move)
        moves.append(move)
    
    # Record the game
    minimax_agent = MinimaxAgent(max_depth=3)
    minimax_agent.name = "Minimax"
    
    mcts_agent = MCTSAgent(iterations=100)
    mcts_agent.name = "MCTS"
    
    pgn_file = recorder.record_game(minimax_agent, mcts_agent, "1-0", moves)
    print(f"Game recorded to {pgn_file}")
    
    # Show the contents of the PGN file
    print("\nPGN file contents:")
    with open(pgn_file, 'r') as f:
        print(f.read())

if __name__ == "__main__":
    print("ChessMindsAI Visualization Demo")
    print("===============================")
    print("This script demonstrates the visualization capabilities of ChessMindsAI.")
    print()
    
    # Demo board display
    demo_board_display()
    
    # Demo game visualization
    try:
        demo_game_visualization()
    except Exception as e:
        print(f"Game visualization demo failed: {e}")
    
    # Demo creating a GIF
    try:
        demo_game_gif()
    except Exception as e:
        print(f"GIF creation demo failed: {e}")
    
    # Demo metrics visualization
    demo_metrics_visualization()
    
    # Demo game recording
    demo_game_recording()