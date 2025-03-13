# ChessEnvironment: A Python Chess Environment for RL Agents

## Overview

This **ChessEnvironment** class is a Python-based chess environment designed to support three different RL-based chess agents:

1. **Minimax with Alpha-Beta Pruning**
2. **Monte Carlo Tree Search (MCTS)**
3. **Neural Network with PPO**

The environment provides methods for **move execution, board evaluation, game state management, and action space mapping**, making it versatile for various reinforcement learning approaches.

---

## Features

- **Move Execution** (`step()`) - Executes a UCI-formatted move and updates the board.
- **State Representation** (`get_state()`) - Provides an 8×8×14 tensor representation for neural networks.
- **Legal Move Retrieval** (`get_legal_moves()`) - Fetches all legal moves.
- **Board Evaluation** (`evaluate()`) - Heuristic-based board scoring for Minimax & MCTS.
- **Move Ordering** (`order_moves()`) - Prioritizes best moves for Minimax efficiency.
- **Cloning Support** (`clone()`) - Allows MCTS simulations without modifying the real board.
- **Action Mapping** (`get_action_space()`, `move_to_action()`, `action_to_move()`) - Converts chess moves into a discrete action space for PPO.

---

## Installation

Ensure you have the required dependencies installed:

```bash
pip install numpy python-chess
```

---

## Usage Guide

### **1. General Initialization**

```python
from chess_environment import ChessEnvironment

env = ChessEnvironment()
env.reset()  # Resets the board to the initial position
```

### **2. Using the Environment for Minimax**

```python
# Retrieve and order legal moves for alpha-beta pruning
ordered_moves = env.order_moves()

# Evaluate the current board position
score = env.evaluate()
```

### **3. Using the Environment for MCTS**

```python
# Clone the environment for safe MCTS simulations
sim_env = env.clone()

# Retrieve legal moves
legal_moves = env.get_legal_moves()

# Simulate a move and check if the game has ended
sim_env.step("e2e4")
if sim_env.is_terminal():
    reward = sim_env.get_reward()
```

### **4. Using the Environment for PPO (Neural Network)**

```python
# Get the board state as an 8x8x14 tensor for input into a CNN
state = env.get_state()

# Convert a move into an action index
move_index = env.move_to_action("e2e4")

# Convert an action index back into a move
move_uci = env.action_to_move(move_index)
```

---

## **Method Reference**

### **Game Management**

- `reset()` - Resets the board.
- `step(move_uci: str) -> (np.ndarray, int, bool)` - Executes a move and returns the new state, reward, and game termination status.
- `get_reward() -> int` - Returns the reward for reinforcement learning.

### **Minimax-Specific Methods**

- `evaluate() -> int` - Returns heuristic evaluation of the board.
- `order_moves() -> list` - Returns a sorted list of best moves for search optimization.

### **MCTS-Specific Methods**

- `clone() -> ChessEnvironment` - Returns a deep copy of the current environment.

### **NeuralNet PPO-Specific Methods**

- `get_state() -> np.ndarray` - Encodes the board into an 8×8×14 tensor.
- `get_action_space() -> dict` - Returns a mapping of action indices to UCI moves.
- `move_to_action(move_uci: str) -> int` - Converts a UCI move to an action index.
- `action_to_move(action_index: int) -> str` - Converts an action index back to a UCI move.

---

## **Conclusion**

This **ChessEnvironment** is designed to be a flexible and efficient tool for implementing Minimax, MCTS, and PPO-based chess agents. It provides a clean API for managing chess game states, evaluating board positions, and integrating RL techniques.

For any questions or contributions, feel free to reach out!
