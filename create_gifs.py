import chess
import chess.pgn
import matplotlib.pyplot as plt
import os
from PIL import Image
import io

def extract_game_from_pgn(pgn_file, game_index=0):
    """Extract a specific game from a PGN file"""
    with open(pgn_file) as f:
        for i in range(game_index + 1):
            game = chess.pgn.read_game(f)
            if game is None:
                return None
            if i == game_index:
                return game
    return None

def create_game_gif(pgn_file, output_file, game_index=0):
    """Create a GIF from a specific game in a PGN file"""
    # Extract the game
    game = extract_game_from_pgn(pgn_file, game_index)
    if not game:
        print(f"Game {game_index} not found in {pgn_file}")
        return
    
    # Get game info
    white = game.headers.get("White", "White")
    black = game.headers.get("Black", "Black")
    
    print(f"Creating GIF for game: {white} vs {black}")
    
    # Set up board and frames
    board = game.board()
    frames = []
    
    # Create a frame for the initial position
    frame = render_board_to_image(board, f"Initial position: {white} vs {black}")
    frames.append(frame)
    
    # Process each move
    move_count = 0
    for move in game.mainline_moves():
        board.push(move)
        move_count += 1
        
        # Create a frame after every move
        title = f"Move {move_count}: {'White' if board.turn == chess.BLACK else 'Black'} played {move.uci()}"
        frame = render_board_to_image(board, title, last_move=move)
        frames.append(frame)
    
    # Save as GIF if we have frames
    if frames:
        try:
            frames[0].save(
                output_file,
                save_all=True,
                append_images=frames[1:],
                duration=1000,  # 1 second per frame
                loop=0
            )
            print(f"GIF saved to {output_file}")
            return True
        except Exception as e:
            print(f"Error saving GIF: {e}")
            return False
    else:
        print("No frames generated!")
        return False

def render_board_to_image(board, title=None, last_move=None):
    """Render a chess board to a PIL Image"""
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    
    # Draw the board
    for row in range(8):
        for col in range(8):
            is_light = (row + col) % 2 == 0
            color = '#F0D9B5' if is_light else '#B58863'
            ax.add_patch(plt.Rectangle((col, 7-row), 1, 1, color=color))
    
    # Add pieces
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
                     fontsize=30, ha='center', va='center', color=color)
    
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
        plt.text(i + 0.5, -0.3, chr(97+i), ha='center', va='center', fontsize=14)
        plt.text(-0.3, i + 0.5, str(i+1), ha='center', va='center', fontsize=14)
    
    plt.xlim(-0.5, 8.5)
    plt.ylim(-0.5, 8.5)
    ax.set_aspect('equal')
    ax.set_axis_off()
    
    if title:
        plt.title(title, fontsize=16)
    
    plt.tight_layout()
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    buf.seek(0)
    return Image.open(buf)

# Create GIFs for the top 3 most decisive games
output_dir = "./visualized_games"
os.makedirs(output_dir, exist_ok=True)

pgn_file = "./results/games/tournament_20250314_172357.pgn"
if os.path.exists(pgn_file):
    print(f"Creating GIFs from {pgn_file}")
    
    # Find decisive games
    decisive_games = []
    with open(pgn_file) as f:
        game_index = 0
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
                
            result = game.headers.get("Result", "*")
            if result != "1/2-1/2":
                white = game.headers.get("White", "Unknown")
                black = game.headers.get("Black", "Unknown")
                decisive_games.append((game_index, white, black, result))
            
            game_index += 1
    
    print(f"Found {len(decisive_games)} decisive games")
    
    # Create GIFs for the first 3 decisive games
    for i, (game_idx, white, black, result) in enumerate(decisive_games[:3]):
        output_file = os.path.join(output_dir, f"game_{i+1}_{white}_vs_{black}.gif")
        create_game_gif(pgn_file, output_file, game_index=game_idx)
else:
    print(f"Error: PGN file {pgn_file} not found")