"""
1. Adapting MCTS for Your Board Game
    • Adapt your MCTS code from Exercise 4 to your chosen board game from Exercise 7.
    • Debug thoroughly to ensure it plays the game correctly and performs well.
"""

from config import SIZE,EMPTY, STRIKE, PLAYER_BLACK, PLAYER_WHITE, DRAW, ITERATIONS_NUMBER
from training_set import TrainingData, TrainingValue
from ex7_gomoku import GomokuGame
from game_gui import GomokuGUI
from typing import List
import numpy as np
import threading
import random
import math
import json


class MCTSNode:

    def __init__(self, game: GomokuGame, parent=None, move=None):
        self.game: GomokuGame = game  # Current game state
        self.parent: MCTSNode = parent  # Parent node
        self.move: tuple = move  # Move that led to this node
        self.children: List[MCTSNode] = []  # List of child nodes
        self.visits = 0  
        self.wins = 0 
 
    def is_fully_expanded(self):
        """Check if all legal moves have been expanded."""
        return len(self.children) == len(self.game.legal_moves())


    def best_child(self, exploration_weight=1):
        """Select the best child node using UCT."""
        def uct_score(child):
            if child.visits == 0:
                return float('inf')  # Encourage exploration of unvisited nodes
            return (child.wins / child.visits) + exploration_weight * math.sqrt(math.log(self.visits) / child.visits)

        max_uct = float('-inf')
        best_children = []

        for child in self.children:
            score = uct_score(child)

            if score > max_uct:
                max_uct = score
                best_children = [child]  # Reset the list with a new best child
            elif score == max_uct:
                best_children.append(child)  # Add to the list of best children
        # print(self.children)
        return random.choice(best_children) 

class MCTSPlayer:
    def __init__(self, exploration_weight=1):
        self.exploration_weight = exploration_weight  # Adjust exploration vs exploitation

    def print_tree(self, node: MCTSNode, level=0, exploration_weight=1.0):
        indent = " " * (4 * level)  # Indentation for tree structure
        
        def uct_score(n):
            if n.visits == 0:
                return float('inf')  # Encourage exploration
            return (n.wins / n.visits) + exploration_weight * math.sqrt(math.log(n.parent.visits) / n.visits) if n.parent else 0

        # Print current node details
        print(f"{indent}Node:{node.move} (Wins: {node.wins}, Visits: {node.visits}, UCT: {uct_score(node):.3f})")

        # Recursively print children
        for child in node.children:
            self.print_tree(child, level + 1, exploration_weight)

    def choose_move(self, game: GomokuGame, iterations: int = ITERATIONS_NUMBER):
        """
        Select the best move using MCTS.
        :param game: A Gomoku object representing the current game state.
        :param iterations: Number of MCTS iterations to perform.
        :return: The best move based on the MCTS algorithm.
        """
        root = MCTSNode(game.clone())  # Initialize the root node with the current game state

        # Perform MCTS iterations
        for _ in range(iterations):
            # rollout
            node = self.selection(root)  # Select a promising node
            child_node = self.expansion(node)

            result = self.simulation(child_node)  # Simulate a random game from the selected node
            
            self.backpropagation(child_node, result)  # Backpropagate the results

        policy_matrix = np.zeros((SIZE, SIZE)) 
        for child in root.children:
            x,y = child.move
            policy_matrix[x, y] = child.visits/iterations
        
        
        training_value = TrainingValue(
            game_state = root.game.encode(),
            policy_target = policy_matrix.flatten()
        )

        best_move = self.find_imamadiate_best_move(node)
        if not best_move:
            best_move = node.best_child(self.exploration_weight).move
            # best_move = max(root.children, key=lambda child: child.visits)
        return best_move, training_value
        
    def find_imamadiate_best_move(self, node):
        cloned_game = node.game.clone()
        legal_moves = cloned_game.legal_moves()
        moves = { m: self.get_next_winner_state(cloned_game, m) for m in legal_moves}
        
        # Check for immediate winning moves
        winning_moves = [m for m, win in moves.items() if win == PLAYER_BLACK]
        if winning_moves:
            move = random.choice(winning_moves)
            return move
       
        # simulate white turn to search for immediate threats
        threats_moves = [m for m in legal_moves if self.search_winning_moves(cloned_game, m)]
        if threats_moves:
            move = random.choice(threats_moves)
            return move

             

    def selection(self, node: MCTSNode) -> MCTSNode: 
        """Select the most promising child node using UCT."""
        while node.is_fully_expanded() and node.children:
            node = node.best_child(self.exploration_weight)  # Pass exploration weight
        return node
    
    def expansion(self, node: MCTSNode):
        """Expand the tree by adding a new child for an untried move."""
        legal_moves = node.game.legal_moves()
        existing_moves = [child.move for child in node.children]
        untried_moves = [m for m in legal_moves if m not in existing_moves]
        move = untried_moves[-1]  # random.choice(untried_moves)
        new_state = node.game.clone()
        new_state.make_move(move)
        child_node = MCTSNode(new_state, node, move)
        node.children.append(child_node)
        return child_node


    def simulation(self, node: MCTSNode):
        cloned_game = node.game.clone() # copy the game state
        while  cloned_game.winner == None:  # Continue until the game ends
            last_move = cloned_game.history[-1]
            if last_move:
                x, y = last_move
                cloned_game.check_winner(x,y)
                if cloned_game.winner: break

            legal_moves = cloned_game.legal_moves()
            moves = { m: self.get_next_winner_state(cloned_game, m) for m in legal_moves}
            
            if cloned_game.turn == PLAYER_BLACK:
                
                # Check for immediate winning moves
                winning_moves = [m for m, win in moves.items() if win == PLAYER_BLACK]
                if winning_moves:
                    move = random.choice(winning_moves)
                    cloned_game.make_move(move)
                    continue

                # simulate white turn to search for immediate threats
                threats_moves = [m for m in legal_moves if self.search_winning_moves(cloned_game, m)]
                if threats_moves:
                    move = random.choice(threats_moves)
                    cloned_game.make_move(move)
                    continue

                

            # Perform a random move if no immediate win
            if legal_moves:
                move = random.choice(legal_moves)
                cloned_game.make_move(move)
            else:
                pass
           
        return cloned_game.winner  # Return the winner of the game

    def get_next_winner_state(self, game: GomokuGame, move):
        cloned_state = game.clone()
        cloned_state.make_move(move)
        return cloned_state.winner  
    
    def search_winning_moves(self, game: GomokuGame, move):
        """Check if a move results in an immediate loss."""
        cloned_state = game.clone()
        cloned_state.turn = PLAYER_WHITE
        cloned_state.make_move(move)
        return cloned_state.winner == PLAYER_WHITE


    def backpropagation(self, node: MCTSNode, winner: int):
        """Propagate the simulation winner back through the tree."""
        while node is not None:
            node.visits += 1  # Increment visit count
            if winner == node.game.turn:
                node.wins += 1
            else: node.wins -= 1 
            
            node = node.parent  # Move up to the parent node



def play_with_mcts(size=SIZE, strike=STRIKE):
    game = GomokuGame(size, strike)  # Initialize the GomoKu game
    mcts_player = MCTSPlayer()  # Create an MCTS player
    # game_gui = GomokuGUI()
    # threading.Thread(target=game_gui.start, daemon=False).start()
    
    training_data: TrainingData = TrainingData()

    while game.winner == None:  # Play until the game ends
        if game.turn == PLAYER_BLACK:  # BLACK is the MCTS player
            # print("MCTS Player (BLACK) is thinking...")
            move, training_value = mcts_player.choose_move(game, iterations=ITERATIONS_NUMBER)  # MCTS chooses the move
            training_data.add_training_value(training_value)
        else:
            # row = int(input("Your move (WHITE, enter row: "))  # Human input
            # col = int(input("Your move (WHITE, enter column: "))  # Human input
            # move = (row, col) # Convert to move format
            # if move not in game.legal_moves():
            #     print("Illegal move. Try again.")
            #     continue

            # Perform a random move for human player
            legal_moves = game.legal_moves()
            if legal_moves:
                move = random.choice(legal_moves)
            else:
                print("Illegal move. Try again.")
                continue
        
        x,y = move

        # game_gui.add_piece(x, y, game.turn)           
        
        game.make_move(move)  # Apply the chosen move and switch turn
        
        # print(game)  # Print the current state of the board

    for training_val in training_data.training_set:
        training_val.set_winner(game.winner)

    # game_gui.mark_winner(x, y, game.winner)

    # Show the final board
    if game.winner == PLAYER_BLACK:
        print("BLACK (MCTS) is the WINNER!")
    elif game.winner == PLAYER_WHITE:
        print("WHITE (Human) is the WINNER!")
    else:
        print("It's a draw!")  
    
    return training_data

def build_training_set():
    games_number = 200
    for i in range(1, games_number):
        print(f"Play [{i}/{games_number}] start")
        training_set_results: TrainingData = play_with_mcts()
        training_set_results.save_to_text_file("training_set_6X6.txt")
    

if __name__ == "__main__": 
    # play_with_mcts()
    build_training_set()