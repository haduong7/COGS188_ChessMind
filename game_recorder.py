import chess
import chess.pgn
import datetime
import io
import os

class GameRecorder:
    def __init__(self, output_dir="./games"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def record_game(self, white_agent, black_agent, game_result, moves, max_moves=200):
        """Record a chess game to a PGN file"""
        game = chess.pgn.Game()
        
        # Set headers
        white_name = getattr(white_agent, 'name', 'White')
        black_name = getattr(black_agent, 'name', 'Black')
        
        game.headers["Event"] = "ChessMindsAI Tournament"
        game.headers["Site"] = "ChessMindsAI Engine"
        game.headers["Date"] = datetime.datetime.now().strftime("%Y.%m.%d")
        game.headers["Round"] = "1"
        game.headers["White"] = white_name
        game.headers["Black"] = black_name
        game.headers["Result"] = game_result
        
        # Add moves
        node = game
        board = chess.Board()
        for move in moves:
            node = node.add_variation(move)
            board.push(move)
            
            # Add annotations for interesting moves
            # (e.g., good moves, blunders, etc. based on engine evaluation)
            
            # Stop if we've reached max moves or the game is over
            if board.is_game_over() or len(game.mainline_moves()) >= max_moves:
                break
        
        # Save the game to a file
        filename = f"{white_name}_vs_{black_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pgn"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, "w") as f:
            f.write(str(game))
        
        return filepath
    
    def record_tournament(self, tournament_results, games, agents):
        """Record all games from a tournament to a PGN file"""
        tournament_file = os.path.join(self.output_dir, f"tournament_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pgn")
        
        with open(tournament_file, "w") as f:
            for game_record in games:
                game = chess.pgn.Game()
                
                # Set headers
                white_idx = agents.index(game_record["white"])
                black_idx = agents.index(game_record["black"])
                
                game.headers["Event"] = "ChessMindsAI Tournament"
                game.headers["Site"] = "ChessMindsAI Engine"
                game.headers["Date"] = datetime.datetime.now().strftime("%Y.%m.%d")
                game.headers["White"] = game_record["white"]
                game.headers["Black"] = game_record["black"]
                game.headers["Result"] = game_record["result"]
                
                # Add moves
                node = game
                for move_uci in game_record["moves"]:
                    move = chess.Move.from_uci(move_uci)
                    node = node.add_variation(move)
                
                f.write(f"{str(game)}\n\n")
        
        return tournament_file
    
    def create_opening_book(self, pgn_files):
        """Create an opening book from a collection of PGN files"""
        openings = {}
        
        for pgn_file in pgn_files:
            with open(pgn_file) as f:
                while True:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break
                    
                    # Extract the first few moves (e.g., 10) for the opening book
                    board = game.board()
                    opening_sequence = []
                    
                    for i, move in enumerate(game.mainline_moves()):
                        if i >= 10:  # Just consider the first 10 moves for the opening
                            break
                        board.push(move)
                        opening_sequence.append(move.uci())
                    
                    # Create a key for this opening sequence
                    key = " ".join(opening_sequence)
                    
                    # Count occurrences of each opening
                    if key in openings:
                        openings[key]["count"] += 1
                        if game.headers["Result"] == "1-0":
                            openings[key]["white_wins"] += 1
                        elif game.headers["Result"] == "0-1":
                            openings[key]["black_wins"] += 1
                        else:
                            openings[key]["draws"] += 1
                    else:
                        openings[key] = {
                            "sequence": opening_sequence,
                            "count": 1,
                            "white_wins": 1 if game.headers["Result"] == "1-0" else 0,
                            "black_wins": 1 if game.headers["Result"] == "0-1" else 0,
                            "draws": 1 if game.headers["Result"] == "1/2-1/2" else 0
                        }
        
        # Save the opening book to a file
        opening_book_file = os.path.join(self.output_dir, "opening_book.json")
        
        import json
        with open(opening_book_file, "w") as f:
            json.dump(openings, f, indent=4)
        
        return opening_book_file