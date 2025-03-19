import chess
import time
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from chess_env import ChessEnvironment

class Tournament:
    def __init__(self, agents, names=None):
        """Initialize tournament with a list of agent objects and their names"""
        self.agents = agents
        self.names = names if names else [f"Agent_{i}" for i in range(len(agents))]
        self.results = {}
        self.game_records = []
    
    def play_game(self, white_idx, black_idx, max_moves=200, time_limit=5):
        """Play a single game between two agents"""
        env = ChessEnvironment()
        env.reset()
        
        white_agent = self.agents[white_idx]
        black_agent = self.agents[black_idx]
        white_name = self.names[white_idx]
        black_name = self.names[black_idx]
        
        move_count = 0
        moves = []
        
        while not env.board.is_game_over() and move_count < max_moves:
            # Get board FEN
            board_fen = env.get_fen()
            
            # Get move from current agent
            if env.board.turn == chess.WHITE:
                move_uci = white_agent.select_move(board_fen, time_limit)
            else:
                move_uci = black_agent.select_move(board_fen, time_limit)
            
            # Apply move
            if move_uci:
                env.step(move_uci)
                moves.append(move_uci)
            else:
                break  # Invalid move
            
            move_count += 1
        
        # Determine result
        if env.board.is_checkmate():
            result = "1-0" if env.board.turn == chess.BLACK else "0-1"
        elif env.board.is_stalemate() or env.board.is_insufficient_material():
            result = "1/2-1/2"
        else:
            result = "1/2-1/2"  # Draw if max moves reached
        
        # Record game
        game_record = {
            "white": white_name,
            "black": black_name,
            "result": result,
            "moves": moves,
            "final_fen": env.get_fen(),
            "move_count": move_count
        }
        
        self.game_records.append(game_record)
        
        return result
    
    def run_tournament(self, games_per_matchup=10, max_moves=200, time_limit=5):
        """Run a round-robin tournament between all agents"""
        # Initialize results matrix
        n_agents = len(self.agents)
        for name in self.names:
            self.results[name] = {"wins": 0, "draws": 0, "losses": 0, "points": 0}
        
        # Play all matchups
        total_games = games_per_matchup * n_agents * (n_agents - 1)
        game_count = 0
        
        for i in range(n_agents):
            for j in range(n_agents):
                if i == j:
                    continue  # Skip self-play
                
                for _ in tqdm(range(games_per_matchup), 
                             desc=f"Games {self.names[i]} (W) vs {self.names[j]} (B)",
                             total=games_per_matchup):
                    result = self.play_game(i, j, max_moves, time_limit)
                    
                    # Update results
                    if result == "1-0":  # White wins
                        self.results[self.names[i]]["wins"] += 1
                        self.results[self.names[j]]["losses"] += 1
                        self.results[self.names[i]]["points"] += 1
                    elif result == "0-1":  # Black wins
                        self.results[self.names[i]]["losses"] += 1
                        self.results[self.names[j]]["wins"] += 1
                        self.results[self.names[j]]["points"] += 1
                    else:  # Draw
                        self.results[self.names[i]]["draws"] += 1
                        self.results[self.names[j]]["draws"] += 1
                        self.results[self.names[i]]["points"] += 0.5
                        self.results[self.names[j]]["points"] += 0.5
                    
                    game_count += 1
        
        return self.results
    
    def get_results_table(self):
        """Get a pandas DataFrame with tournament results"""
        results_list = []
        for name, stats in self.results.items():
            total_games = stats["wins"] + stats["draws"] + stats["losses"]
            win_rate = stats["wins"] / total_games if total_games > 0 else 0
            
            results_list.append({
                "Agent": name,
                "Wins": stats["wins"],
                "Draws": stats["draws"],
                "Losses": stats["losses"],
                "Points": stats["points"],
                "Win Rate": f"{win_rate:.2%}"
            })
        
        return pd.DataFrame(results_list).sort_values("Points", ascending=False)
    
    def plot_results(self):
        """Create a bar chart of tournament results"""
        df = self.get_results_table()
        
        plt.figure(figsize=(10, 6))
        x = range(len(df))
        width = 0.25
        
        plt.bar(x, df["Wins"], width, label="Wins", color="green")
        plt.bar([i + width for i in x], df["Draws"], width, label="Draws", color="gray")
        plt.bar([i + width*2 for i in x], df["Losses"], width, label="Losses", color="red")
        
        plt.xlabel("Agents")
        plt.ylabel("Games")
        plt.title("Tournament Results")
        plt.xticks([i + width for i in x], df["Agent"])
        plt.legend()
        
        plt.tight_layout()
        return plt