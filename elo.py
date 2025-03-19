
import random

def calculate_elo(winner_elo, loser_elo, K=32, draw=False):
    """Updates Elo ratings after a match."""
    expected_winner = 1 / (1 + 10 ** ((loser_elo - winner_elo) / 400))
    expected_loser = 1 - expected_winner  # Expected score for the loser

    if draw:
        winner_score, loser_score = 0.5, 0.5
    else:
        winner_score, loser_score = 1, 0

    new_winner_elo = winner_elo + K * (winner_score - expected_winner)
    new_loser_elo = loser_elo + K * (loser_score - expected_loser)

    return round(new_winner_elo, 2), round(new_loser_elo, 2)


def evaluate_models(current_model, best_model, num_games=10, play_game_fn=None):
    print(f"Iteration {iteration + 1}: Action Space Size: {len(action_space)}")
    """Plays games between the current model and the best model to estimate Elo.

    Args:
        current_model: The latest trained model.
        best_model: The best model found so far.
        num_games: Number of matches to play.
        play_game_fn: Function to simulate a game (should return "current", "best", or "draw").

    Returns:
        Tuple (current_wins, best_wins, draws)
    """
    if play_game_fn is None:
        raise ValueError("You must provide a function to simulate games.")

    current_wins = 0
    best_wins = 0
    draws = 0

    for _ in range(num_games):
        result = play_game_fn(current_model, best_model)  # Function should return "current", "best", or "draw"
        if result == "current":
            current_wins += 1
        elif result == "best":
            best_wins += 1
        else:
            draws += 1

    return current_wins, best_wins, draws