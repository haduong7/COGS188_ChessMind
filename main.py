import os
import torch
import argparse
import matplotlib.pyplot as plt

from minimax_agent import MinimaxAgent
from mcts_agent import MCTSAgent
from neural_agent import NeuralAgent, ChessNN
from hybrid_agent import HybridAgent
from ppo_trainer import PPOTrainer
from tournament import Tournament
from visualization import display_board, visualize_game, create_game_gif
from metrics_visualizer import MetricsVisualizer
from game_recorder import GameRecorder

def train_neural_agent(args):
    """Train the neural network agent"""
    print("Starting neural network training...")
    
    # Create model and trainer
    model = ChessNN()
    trainer = PPOTrainer(model=model, lr=args.learning_rate, gamma=args.gamma)
    
    # Train model
    losses = trainer.train(
        iterations=args.iterations,
        games_per_iteration=args.games_per_iteration,
        save_path=os.path.join(args.model_dir, "chess_model")
    )
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("PPO Training Loss")
    plt.savefig(os.path.join(args.model_dir, "training_loss.png"))
    
    print(f"Model saved to {args.model_dir}")
    rewards = losses 
    return model, rewards

def run_tournament(args, model=None):
    """Run a tournament between different agents"""
    # Create agents
    minimax_agent = MinimaxAgent(max_depth=args.minimax_depth)
    minimax_agent.name = "Minimax"
    
    mcts_agent = MCTSAgent(iterations=args.mcts_iterations)
    mcts_agent.name = "MCTS"
    
    # Load or use provided neural network model
    neural_agent = NeuralAgent(model_path=args.model_path if model is None else None)
    if model is not None:
        neural_agent.model = model
    neural_agent.name = "Neural"
    
    # Create hybrid agent
    hybrid_agent = HybridAgent(neural_agent, iterations=args.hybrid_iterations)
    hybrid_agent.name = "Hybrid"
    
    # Set up tournament
    agents = [minimax_agent, mcts_agent, neural_agent, hybrid_agent]
    names = ["Minimax", "MCTS", "Neural", "Hybrid"]
    
    tournament = Tournament(agents, names)
    
    # Run tournament
    results = tournament.run_tournament(
        games_per_matchup=args.games_per_matchup,
        max_moves=args.max_moves,
        time_limit=args.time_limit
    )
    
    # Display and save results
    results_table = tournament.get_results_table()
    print("\nTournament Results:")
    print(results_table)
    
    # Plot results
    plt_fig = tournament.plot_results()
    plt_fig.savefig(os.path.join(args.output_dir, "tournament_results.png"))
    
    # Save results to CSV
    results_table.to_csv(os.path.join(args.output_dir, "tournament_results.csv"), index=False)
    
    print(f"Results saved to {args.output_dir}")
    
    return results_table, tournament, agents, names

def main():
    parser = argparse.ArgumentParser(description="ChessMindsAI - Comparison of Chess AI Approaches")
    
    # General arguments
    parser.add_argument("--mode", type=str, choices=["train", "tournament", "both"], default="both",
                        help="Run mode: train, tournament, or both")
    parser.add_argument("--model_dir", type=str, default="./models",
                        help="Directory to save/load models")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save results")
    
    # Training arguments
    parser.add_argument("--iterations", type=int, default=50,
                        help="Number of training iterations")
    parser.add_argument("--games_per_iteration", type=int, default=10,
                        help="Number of self-play games per iteration")
    parser.add_argument("--learning_rate", type=float, default=0.0001,
                        help="Learning rate for neural network training")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor for rewards")
    
    # Tournament arguments
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to pre-trained model (if not training)")
    parser.add_argument("--games_per_matchup", type=int, default=10,
                        help="Number of games per agent matchup")
    parser.add_argument("--max_moves", type=int, default=200,
                        help="Maximum moves per game")
    parser.add_argument("--time_limit", type=float, default=5,
                        help="Time limit per move in seconds")
    parser.add_argument("--minimax_depth", type=int, default=4,
                        help="Search depth for minimax agent")
    parser.add_argument("--mcts_iterations", type=int, default=1000,
                        help="Iterations for MCTS agent")
    parser.add_argument("--hybrid_iterations", type=int, default=400,
                        help="Iterations for hybrid agent")
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    model = None
    rewards = None
    tournament_results = None
    tournament_obj = None
    agents = None
    agent_names = None
    
    # Run in selected mode
    if args.mode == "train" or args.mode == "both":
        model, rewards = train_neural_agent(args)
    
    if args.mode == "tournament" or args.mode == "both":
        tournament_results, tournament_obj, agents, agent_names = run_tournament(args, model)
    
    # Initialize visualizers
    metrics_viz = MetricsVisualizer(args.output_dir)
    game_recorder = GameRecorder(os.path.join(args.output_dir, "games"))
    
    # Visualize training progress only if rewards are available
    if args.mode in ["train", "both"] and rewards is not None:
        metrics_viz.plot_training_progress(rewards, "Neural Network (PPO)")
    
    # Visualize tournament results only if tournament was run
    if args.mode in ["tournament", "both"] and tournament_results is not None:
        metrics_viz.plot_tournament_results(tournament_results)
        
        # Record tournament games
        if tournament_obj and tournament_obj.game_records:
            game_recorder.record_tournament(tournament_results, tournament_obj.game_records, agent_names)
        
        # Visualize a match between the best performing agents
        if agents and len(agents) > 1:
            print("\nVisualizing a match between top agents...")
            try:
                # Find top two agents
                top_agent = agents[0]  # Default to first if we can't determine
                runner_up = agents[1]  # Default to second
                
                if tournament_results is not None:
                    top_index = tournament_results['Points'].idxmax()
                    top_agent = agents[top_index]
                    # Find second best
                    remaining_df = tournament_results.drop(top_index)
                    if not remaining_df.empty:
                        runner_up_index = remaining_df['Points'].idxmax()
                        runner_up = agents[runner_up_index]
                
                board, moves = visualize_game(top_agent, runner_up, delay=0.5)
                
                # Create a GIF of the game
                gif_path = create_game_gif(top_agent, runner_up, 
                                          filename=os.path.join(args.output_dir, "top_match.gif"))
                print(f"Game recording saved as {gif_path}")
            except Exception as e:
                print(f"Error visualizing game: {e}")
        
        # Plot computational efficiency if available
        if agents:
            efficiency_data = []
            for agent in agents:
                # Only measure if the agent has this capability
                if hasattr(agent, 'get_stats'):
                    stats = agent.get_stats()
                    efficiency_data.append(stats)
            
            if efficiency_data:
                nodes_per_second = [stats.get('nodes_per_second', 0) for stats in efficiency_data]
                time_per_move = [stats.get('time_taken', 0) for stats in efficiency_data]
                metrics_viz.plot_computational_efficiency(agent_names, nodes_per_second, time_per_move)

if __name__ == "__main__":
    main()