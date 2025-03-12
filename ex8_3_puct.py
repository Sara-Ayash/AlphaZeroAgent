"""
3. Implementing PUCT Algorithm
    • Write PUCTNode and PUCTPlayer classes by adapting MCTSNode and MCTSPlayer.
    • Key changes:
        - Replace random rollouts with evaluations from the neural network.
        - Each node maintains a prior probability P from the policy head of its parent.
        - Each node maintains Q (average value) and N (visit count). Avoid calling it wins.
        - Use the PUCT scoring formula: U (s, a) = Q(s, a) + cpuct· P (s, a)·N (s) 1 + N (s, a)
"""

import torch
from config import SIZE,EMPTY, STRIKE, PLAYER_BLACK, PLAYER_WHITE, DRAW, ITERATIONS_NUMBER
from ex7_gomoku import GomokuGame
from ex8_2_neural_network import GomokuNetwork
from game_gui import GomokuGUI
from typing import List
import numpy as np
import random
import math


class PUCTNode:

    def __init__(self, game: GomokuGame, parent=None, move=None, prior=1.0):
        self.game = game  # Current game state
        self.parent: PUCTNode = parent  # Parent node
        self.move: tuple = move  # Move that led to this node
        self.children: List[PUCTNode] = []  # List of child nodes
        self.P = prior  # Prior probability (P) from the neural network
        self.Q = 0  # Average value of this node
        self.N = 0  # Visit count


    def is_fully_expanded(self):
        """Check if all legal moves have been expanded."""
        return len(self.children) == len(self.game.legal_moves())
    

    def best_child(self, cpuct):
        """Select the best child node using PUCT."""
        def puct_score(child):
            if child.N == 0:
                return float('inf')  # Encourage unvisited nodes
            U = child.Q + cpuct * child.P * math.sqrt(self.N) / (1 + child.N)
            return U
        
        max_puct = float('-inf') 
        best_children = []

        for child in self.children:
            score = puct_score(child)

            if score > max_puct:
                max_puct = score
                best_children = [child]  # Reset the list with a new best child
            elif score == max_puct:
                best_children.append(child)  # Add to the list of best children

        return random.choice(best_children) 



class PUCTPlayer:
    def __init__(self, network, simulations=100, cpuct=1.5):
        self.network = network
        self.simulations = simulations
        self.cpuct = cpuct

    def choose_move(self, game, iterations=ITERATIONS_NUMBER):
        """Perform PUCT search and return the best move."""
        root = PUCTNode(game)


        # Perform iterations
        for _ in range(iterations):     
            node = self.selection(root)  # Select a promising node
            node = self.expansion(node)
            self.backpropagation(node)  # Backpropagate the results
 

    def selection(self, node: PUCTNode) -> PUCTNode: 
        """Select the most promising child node using PUCT."""
        while node.is_fully_expanded() and node.children:
            node = node.best_child(self.cpuct) 
        return node


    def expansion(self, node: PUCTNode) -> PUCTNode:
        encoded_state = torch.tensor(node.game.encode(), dtype=torch.float32).unsqueeze(0)
        encoded_state_mtx = encoded_state.view(-1, 3, SIZE, SIZE)  # Reshape to (batch_size, 3, 9, 9)

        P_pred, Q_pred = self.network(encoded_state_mtx)

        node.Q = Q_pred

        untried_moves = [move for move in node.game.legal_moves() if move not in [child.move for child in node.children]]
        policy_matrix = P_pred.view(-1, 3, SIZE, SIZE)
        # normalize
 
        for move in untried_moves:
            x,y = move
            child_node = PUCTNode(new_state, parent=node, move=move, prior=(policy_matrix[x, y])) 
            node.children.append(child_node)  # Add the new child node to the tree
        
        return node # Return the newly expanded node
    
    
    def backpropagation(self, node: PUCTNode):

        """Propagate the simulation result back through the tree."""
        while node is not None:
            node.N += 1 
            total_Q = sum(child.Q for child in node.children)
            avarage_Q = total_Q/len(node.children) if node.children and total_Q else node.Q
            node.Q = avarage_Q * node.game.turn
            node = node.parent  # Move up to the parent node


    def print_tree(self, node: PUCTNode, level=0):
        indent = " " * (4 * level)  # Indentation for tree structure

        def puct_score(child):
            if child.N == 0:
                return float('inf')  # Encourage unvisited nodes
            U = child.Q + self.cpuct * child.P * math.sqrt(self.N) / (1 + child.N)
            return U

        # Print current node details
        print(f"{indent}Node:{node.move} (Q: {node.Q}, P: {node.P}, N: {node.N}, PUCT: {puct_score(node):.3f})")

        # Recursively print children
        for child in node.children:
            self.print_tree(child, level + 1, self.cpuct)



def play_with_puct(size=SIZE, strike=STRIKE):

    game = GomokuGame(size, strike)  # Initialize the GomoKu game
    network = GomokuNetwork(size)
    puct_player = PUCTPlayer()  # Create an PUCT player
    game_gui = GomokuGUI()
    threading.Thread(target=game_gui.start, daemon=False).start()
    

    while game.winner is None:  # Play until the game ends
        if game.turn == PLAYER_BLACK:  # BLACK is the PUCT player
            print("PUCT Player (BLACK) is thinking...")
            move, training_value = puct_player.choose_move(game)  # PUCT chooses the move
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
        game_gui.add_piece(x, y, game.turn)           
        game.make_move(move)  # Apply the chosen move and switch turn
        
    print(game)  # Print the current state of the board

    game_gui.mark_winner(x, y, game.winner)

    # Show the final board
    if game.winner == PLAYER_BLACK:
        print("BLACK (PUCT) is the WINNER!")
    elif game.winner == PLAYER_WHITE:
        print("WHITE (Human) is the WINNER!")
    else:
        print("It's a draw!")  
        

if __name__ == "__main__": 
    play_with_puct()










































# import math
# import numpy as np
# import torch
# from ex7_gomoku import GomokuGame
# from ex8_neural_network import GameNetwork


# class PUCTNode:
#     def __init__(self, parent=None, prior=1.0):
#         """
#         Represents a node in the MCTS tree using PUCT.

#         Parameters:
#         - parent (PUCTNode): Parent node in the tree.
#         - prior (float): Prior probability of selecting this move (from policy network).
#         """
#         self.parent = parent
#         self.children = {}  # Dictionary {move: PUCTNode}
#         self.N = 0  # Visit count
#         self.W = 0  # Total value sum
#         self.Q = 0  # Average value
#         self.P = prior  # Prior probability from policy network

#     def select_child(self, c_puct=1.0):
#         """
#         Selects the child node with the highest PUCT value.
#         """
#         best_move, best_child = max(
#             self.children.items(),
#             key=lambda item: item[1].Q + c_puct * item[1].P * math.sqrt(self.N) / (1 + item[1].N)
#         )
#         return best_move, best_child

#     def expand(self, game, policy_probs):
#         """
#         Expands the node by adding new child nodes.
#         """
#         for move, prob in enumerate(policy_probs):
#             if move in game.legal_moves():
#                 self.children[move] = PUCTNode(parent=self, prior=prob)

#     def update(self, value):
#         """
#         Updates the node with a new value.
#         """
#         self.N += 1
#         self.W += value
#         self.Q = self.W / self.N  # Update average value



# class PUCTPlayer:
#     def __init__(self, network, c_puct=1.0, simulations=100):
#         """
#         MCTS-based player using PUCT and a neural network.

#         Parameters:
#         - network (GameNetwork): Neural network guiding MCTS.
#         - c_puct (float): Exploration parameter for PUCT.
#         - simulations (int): Number of MCTS simulations per move.
#         """
#         self.network = network
#         self.c_puct = c_puct
#         self.simulations = simulations

#     def search(self, game):
#         """
#         Performs MCTS search using PUCT and the neural network.
#         """
#         root = PUCTNode(parent=None, prior=1.0)

#         for _ in range(self.simulations):
#             node, path = root, []
#             game_copy = game.clone()

#             # Selection and expansion
#             while node.children:
#                 move, node = node.select_child(self.c_puct)
#                 game_copy.make_move(move)
#                 path.append(node)

#             # Evaluation using the neural network
#             encoded_state = game_copy.encode()
#             policy, value = self.network(encoded_state.unsqueeze(0))

#             # Expand the node
#             node.expand(game_copy, policy.detach().numpy().flatten())

#             # Backpropagation: Update all nodes along the path
#             for node in reversed(path):
#                 node.update(value.item())

#         # Choose the move with the highest visit count
#         return max(root.children.items(), key=lambda item: item[1].N)[0]
    

# if __name__ == "__main__":
#     # Initialize board size
#     board_size = 9  # Can be 9x9 or 15x15

#     # Create the neural network
#     cnn_model = GameNetwork(board_size)
#     cnn_model.load("game_network.pth")  # Load pre-trained model (if available)

#     # Create a PUCT-based player
#     puct_player = PUCTPlayer(network=cnn_model, simulations=100)

#     # Create a game instance
#     game = GomokuGame()  

#     # Main loop to test PUCT search
#     while not game.is_over():
#         print(game)  # Print current board state
#         move = puct_player.search(game)  # Use PUCT to choose the best move
#         print(f"PUCT selected move: {move}")

#         game.make_move(move)  # Apply the move

#     # Print the result
#     print("Game Over. Winner:", game.get_winner())

