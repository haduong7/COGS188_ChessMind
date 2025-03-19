import berserk
import chess
import time
import torch
from neural_agent import ChessNN, NeuralAgent
from hybrid_agent import HybridAgent
from chess_env import ChessEnvironment

# Initialize the Lichess API client
session = berserk.TokenSession(API_TOKEN)
client = berserk.Client(session)

# Initialize chess environment
env = ChessEnvironment()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load nn model
model = ChessNN().to(device)
model.load_state_dict(torch.load("models/chess_model_best.pt", map_location=device))
model.eval()

# agent init
neural_agent = NeuralAgent(model_path="models/chess_model_best.pt")
hybrid_agent = HybridAgent(neural_agent, iterations=400)

def get_model_move(board):
    """Uses the PPO model to determine the best move."""
    env.set_fen(board.fen())  # Sync environment with board state
    state_tensor = torch.FloatTensor(env.get_state()).unsqueeze(0).to(device) 

    with torch.no_grad():
        policy_logits, _ = model(state_tensor)
        policy = torch.softmax(policy_logits, dim=1).squeeze(0)

    # Choose best legal move
    legal_moves = list(board.legal_moves)
    move_probs = {m.uci(): policy[i] for i, m in enumerate(legal_moves)}

    best_move = max(move_probs, key=move_probs.get)
    print(f"Model selected move: {best_move}")
    return best_move

def get_hybrid_move(board):
    """Uses the Hybrid agent to determine the best move."""
    best_move = hybrid_agent.select_move(board.fen(), time_limit=5.0)
    print(f"Hybrid agent selected move: {best_move}")
    return best_move

def save_pgn(game_id):
    """Fetches and saves the PGN of a finished game."""
    try:
        pgn = client.games.export(game_id, as_pgn=True)
        pgn_file = f"games/{game_id}.pgn"
        with open(pgn_file, "w") as file:
            file.write(pgn)
        print(f"PGN saved as {pgn_file}")
    except Exception as e:
        print(f"Failed to fetch PGN: {e}")

def play_game(level, model_type="hybrid"):
    """
    Challenges a Lichess AI at a given level and plays using the selected model.
    
    model_type: "nn" for NN-only agent, "hybrid" for the Hybrid agent.
    """
    global client, env, device

    # Start the AI challenge
    game_info = client.challenges.create_ai(level=level, clock_limit=180, clock_increment=2)
    game_id = game_info.get("id")
    if not game_id:
        raise ValueError("Lichess AI game creation failed!")
    
    print(f"Game started: https://lichess.org/{game_id}")
    
    game_stream = client.bots.stream_game_state(game_id)
    board = chess.Board()
    bot_color = None
    
    for event in game_stream:
        print(f"Event received: {event}")  # DEBUG
        
        if event["type"] == "gameFull":
            bot_id = client.account.get()["id"]
            if "id" in event["white"] and event["white"]["id"] == bot_id:
                bot_color = chess.WHITE
            else:
                bot_color = chess.BLACK
            print(f"Bot is playing as: {'White' if bot_color == chess.WHITE else 'Black'}")
            board = chess.Board()  # Reset board
            
            # If bot is White, make first move immediately
            if bot_color == chess.WHITE:
                print("Bot is White, making first move immediately...")
                if model_type == "nn":
                    best_move = get_model_move(board)
                else:
                    best_move = get_hybrid_move(board)
                
                if best_move:
                    print(f"First move: {best_move}")
                    response = client.bots.make_move(game_id, best_move)
                    print(f"Move response: {response}")
                else:
                    print("ERROR: Failed to select a move for the first turn!")
        
        if event["type"] == "gameState":
            moves = event.get("moves", "").split()
            board = chess.Board()
            for move in moves:
                board.push_uci(move)
            print(f"Current board position:\n{board}")  # DEBUG
            
            if board.is_game_over():
                result = board.result()
                print(f"Game Over! Result: {result}")

                save_pgn(game_id)
                
                return
            
            if board.turn == bot_color:
                print("Bot's turn to move...")  # DEBUG
                if model_type == "nn":
                    best_move = get_model_move(board)
                else:
                    best_move = get_hybrid_move(board)
                
                print(f"Bot plays: {best_move}")  # DEBUG
                response = client.bots.make_move(game_id, best_move)
                print(f"Move response: {response}")  # DEBUG
                time.sleep(1)  # Adjust delay if necessary

# Example usage:
# To use the NN-only agent:
# play_game(level=2, model_type="nn")
# To use the Hybrid agent:
play_game(level=1, model_type="nn")