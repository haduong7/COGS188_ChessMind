"""
Visualization tools for ChessMindsAI - Matplotlib-based version
"""
import chess
import matplotlib.pyplot as plt
import numpy as np
import time
import io
from PIL import Image
from IPython.display import clear_output, display

def display_board(board, title=None, last_move=None):
    """Display a chess board using matplotlib"""
    # Create a figure and axis
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
    plt.show()
    
    return plt

def visualize_game(white_agent, black_agent, max_moves=100, delay=0.5):
    """Play and visualize a game between two agents"""
    board = chess.Board()
    moves = []
    
    white_name = getattr(white_agent, 'name', 'White')
    black_name = getattr(black_agent, 'name', 'Black')
    
    move_count = 0
    
    while not board.is_game_over() and move_count < max_moves:
        # Display the current board
        if board.turn == chess.WHITE:
            title = f"Move {move_count+1}: {white_name} (White) thinking..."
        else:
            title = f"Move {move_count+1}: {black_name} (Black) thinking..."
            
        display_board(board, title)
        
        # Get the move from the current player
        start_time = time.time()
        if board.turn == chess.WHITE:
            move_uci = white_agent.select_move(board.fen())
        else:
            move_uci = black_agent.select_move(board.fen())
        think_time = time.time() - start_time
        
        # Convert UCI to chess.Move
        move = chess.Move.from_uci(move_uci)
        moves.append(move)
        
        # Make the move
        board.push(move)
        move_count += 1
        
        # Display the updated board
        if board.turn == chess.WHITE:
            title = f"Move {move_count}: {black_name} played {move_uci} ({think_time:.2f}s)"
        else:
            title = f"Move {move_count}: {white_name} played {move_uci} ({think_time:.2f}s)"
        
        display_board(board, title, last_move=move)
        time.sleep(delay)  # Add delay to better see the moves
        clear_output(wait=True)
    
    # Display final position
    result = board.result()
    if result == "1-0":
        title = f"Game over: {white_name} (White) wins"
    elif result == "0-1":
        title = f"Game over: {black_name} (Black) wins"
    else:
        title = "Game over: Draw"
    
    display_board(board, title)
    print(f"Game finished in {move_count} moves")
    print(f"Result: {result}")
    
    return board, moves

def create_game_gif(white_agent, black_agent, filename="chess_game.gif", max_moves=100, frame_duration=500):
    """Create an animated GIF of a game between two agents using matplotlib"""
    board = chess.Board()
    frames = []
    
    move_count = 0
    last_move = None
    
    while not board.is_game_over() and move_count < max_moves:
        # Get the move from the current player
        if board.turn == chess.WHITE:
            move_uci = white_agent.select_move(board.fen())
        else:
            move_uci = black_agent.select_move(board.fen())
        
        # Convert UCI to chess.Move
        move = chess.Move.from_uci(move_uci)
        
        # Make the move
        board.push(move)
        move_count += 1
        
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
        
        last_move = move
        
        # Add board labels
        for i in range(8):
            plt.text(i + 0.5, -0.3, chr(97+i), ha='center', va='center')
            plt.text(-0.3, i + 0.5, str(i+1), ha='center', va='center')
        
        # Setup the plot
        plt.xlim(0, 8)
        plt.ylim(0, 8)
        ax.set_aspect('equal')
        ax.set_axis_off()
        
        title = f"Move {move_count}: {'White' if not board.turn else 'Black'} played {move_uci}"
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
            filename,
            save_all=True,
            append_images=frames[1:],
            duration=frame_duration,
            loop=0
        )
    
    return filename