"""
pgn_visualizer.py - Visualize a game from an existing PGN file
"""
import chess
import chess.pgn
import matplotlib.pyplot as plt
import time
import os
import io
from PIL import Image

def display_board(board, title=None, last_move=None, save_path=None):
    """Display a chess board using matplotlib"""
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    
    # Draw the board
    for row in range(8):
        for col in range(8):
            is_light = (row + col) % 2 == 0
            color = '#F0D9B5' if is_light else '#B58863'  # Light/dark brown
            ax.add_patch(plt.Rectangle((col, 7-row), 1, 1, color=color))
    
    # Add piece symbols
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            file_idx = chess.square_file(square)
            rank_idx = chess.square_rank(square)
            
            symbol = piece.symbol()
            # Map piece symbols to more readable versions
            piece_symbols = {
                'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
                'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
            }
            color = 'white' if piece.color == chess.WHITE else 'black'
            plt.text(file_idx + 0.5, rank_idx + 0.5, piece_symbols.get(symbol, symbol), 
                     fontsize=24, ha='center', va='center', color=color)
    
    # Highlight last move
    if last_move:
        from_file = chess.square_file(last_move.from_square)
        from_rank = chess.square_rank(last_move.from_square)
        to_file = chess.square_file(last_move.to_square)
        to_rank = chess.square_rank(last_move.to_square)
        
        # Highlight source and destination squares
        ax.add_patch(plt.Rectangle((from_file, from_rank), 1, 1, color='#aaffaa', alpha=0.5))
        ax.add_patch(plt.Rectangle((to_file, to_rank), 1, 1, color='#aaffaa', alpha=0.5))
    
    # Add board labels
    for i in range(8):
        # File labels (a-h)
        plt.text(i + 0.5, -0.3, chr(97+i), ha='center', va='center')
        # Rank labels (1-8)
        plt.text(-0.3, i + 0.5, str(i+1), ha='center', va='center')
    
    # Setup the plot
    plt.xlim(0, 8)
    plt.ylim(0, 8)
    ax.set_aspect('equal')
    ax.set_axis_off()
    
    if title:
        plt.title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
        return save_path
    else:
        plt.show()
        plt.close()

def create_game_gif_from_pgn(pgn_file, output_file="chess_game.gif", max_moves=30, frame_duration=500):
    """Create a GIF from a PGN file"""
    # Open the PGN file
    with open(pgn_file) as f:
        game = chess.pgn.read_game(f)
    
    if not game:
        print("No game found in PGN file")
        return None
    
    # Set up the initial board
    board = game.board()
    frames = []
    
    # Get the player names
    white = game.headers.get("White", "White")
    black = game.headers.get("Black", "Black")
    
    # Process each move in the game
    move_count = 0
    last_move = None
    
    for move in game.mainline_moves():
        # Make the move
        board.push(move)
        last_move = move
        move_count += 1
        
        if move_count > max_moves:
            break
        
        # Create a figure for the current board state
        plt.figure(figsize=(8, 8))
        ax = plt.gca()
        
        # Draw the board
        for row in range(8):
            for col in range(8):
                is_light = (row + col) % 2 == 0
                color = '#F0D9B5' if is_light else '#B58863'  # Light/dark brown
                ax.add_patch(plt.Rectangle((col, 7-row), 1, 1, color=color))
        
        # Add piece symbols
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                file_idx = chess.square_file(square)
                rank_idx = chess.square_rank(square)
                
                symbol = piece.symbol()
                piece_symbols = {
                    'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
                    'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
                }
                color = 'white' if piece.color == chess.WHITE else 'black'
                plt.text(file_idx + 0.5, rank_idx + 0.5, piece_symbols.get(symbol, symbol), 
                         fontsize=24, ha='center', va='center', color=color)
        
        # Highlight last move
        if last_move:
            from_file = chess.square_file(last_move.from_square)
            from_rank = chess.square_rank(last_move.from_square)
            to_file = chess.square_file(last_move.to_square)
            to_rank = chess.square_rank(last_move.to_square)
            
            ax.add_patch(plt.Rectangle((from_file, from_rank), 1, 1, color='#aaffaa', alpha=0.5))
            ax.add_patch(plt.Rectangle((to_file, to_rank), 1, 1, color='#aaffaa', alpha=0.5))
        
        # Add board labels
        for i in range(8):
            plt.text(i + 0.5, -0.3, chr(97+i), ha='center', va='center')
            plt.text(-0.3, i + 0.5, str(i+1), ha='center', va='center')
        
        # Setup the plot
        plt.xlim(0, 8)
        plt.ylim(0, 8)
        ax.set_aspect('equal')
        ax.set_axis_off()
        
        move_str = move.uci()
        turn = "Black" if board.turn == chess.WHITE else "White"  # Player who just moved
        title = f"Move {move_count}: {turn} played {move_str}"
        plt.title(title)
        
        plt.tight_layout()
        
        # Save figure to a PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        frames.append(img)
        plt.close()
    
    # Save the frames as an animated GIF
    if frames:
        frames[0].save(
            output_file,
            save_all=True,
            append_images=frames[1:],
            duration=frame_duration,
            loop=0
        )
        print(f"GIF created: {output_file}")
    
    return output_file

def visualize_interesting_games(pgn_file, output_dir="./visualized_games"):
    """Find and visualize the most interesting games from a tournament PGN file"""
    os.makedirs(output_dir, exist_ok=True)
    
    # List to store game metadata
    games = []
    
    # Read all games from PGN
    with open(pgn_file) as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
                
            white = game.headers.get("White", "Unknown")
            black = game.headers.get("Black", "Unknown")
            result = game.headers.get("Result", "*")
            
            # Count moves
            move_count = len(list(game.mainline_moves()))
            
            games.append({
                "white": white,
                "black": black,
                "result": result,
                "moves": move_count,
                "game": game
            })
    
    if not games:
        print("No games found in PGN file")
        return
    
    print(f"Found {len(games)} games in {pgn_file}")
    
    # Find non-draw games (usually more interesting)
    decisive_games = [g for g in games if g["result"] != "1/2-1/2"]
    
    # If no decisive games, sort by move count (longer games might be more interesting)
    if not decisive_games:
        games.sort(key=lambda g: g["moves"], reverse=True)
        selected_games = games[:3]  # Take top 3 longest games
    else:
        selected_games = decisive_games[:3]  # Take top 3 decisive games
    
    # Visualize selected games
    for i, game_data in enumerate(selected_games):
        game = game_data["game"]
        white = game_data["white"]
        black = game_data["black"]
        result = game_data["result"]
        
        print(f"Visualizing game {i+1}: {white} vs {black}, Result: {result}")
        
        # Create GIF
        gif_file = os.path.join(output_dir, f"game_{i+1}_{white}_vs_{black}.gif")
        create_game_gif_from_pgn(pgn_file, gif_file, max_moves=30, frame_duration=500)
        
        # Create a snapshot of the final position
        board = game.board()
        for move in game.mainline_moves():
            board.push(move)
        
        final_position_file = os.path.join(output_dir, f"final_position_{i+1}_{white}_vs_{black}.png")
        display_board(board, f"Final Position: {white} vs {black}, Result: {result}", save_path=final_position_file)
    
    print(f"Visualization complete. Files saved to {output_dir}")

if __name__ == "__main__":
    # Path to your tournament PGN file
    pgn_file = "games/I4IaYeKM.pgn"
    
    # Visualize interesting games
    visualize_interesting_games(pgn_file)