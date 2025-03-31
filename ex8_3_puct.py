"""
3. Implementing PUCT Algorithm
    • Write PUCTNode and PUCTPlayer classes by adapting MCTSNode and MCTSPlayer.
    • Key changes:
        - Replace random rollouts with evaluations from the neural network.
        - Each node maintains a prior probability P from the policy head of its parent.
        - Each node maintains Q (average value) and N (visit count). Avoid calling it wins.
        - Use the PUCT scoring formula: U (s, a) = Q(s, a) + cpuct· P (s, a)·N (s) 1 + N (s, a)
"""

import threading
import torch
from config import SIZE, STRIKE, PLAYER_BLACK, PLAYER_WHITE, ITERATIONS_NUMBER
from ex7_gomoku import GomokuGame
from ex8_1_mcts import MCTSPlayer
from ex8_2_neural_network import GameNetwork
from game_gui import GomokuGUI
from typing import List
import numpy as np
import random
import math


class PUCTNode:

    def __init__(self, game: GomokuGame, parent=None, move=None, prior=0.0):
        self.game = game  # Current game state
        self.parent: PUCTNode = parent  # Parent node
        self.move: tuple = move  # Move that led to this node
        self.children: List[PUCTNode] = []  # List of child nodes
        self.P = prior  # Prior probability (P) from the neural network
        self.Q = 0  # Average value of this node
        self.N = 0  # Visit count
        self.total_value = 0.0  


    def is_fully_expanded(self):
        """Check if all legal moves have been expanded."""
        return len(self.children) == len(self.game.legal_moves())
    

    def best_child(self, cpuct=1.0):
        """Select the best child node using PUCT."""
        def puct_score(child):
            if child.N == 0:
                return float('inf')  # Encourage unvisited nodes

            return child.Q * self.game.turn + cpuct * child.P * math.sqrt(self.N) / (1 + child.N)
        
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
         
    def choose_move(self, game: GomokuGame, iterations=ITERATIONS_NUMBER):
        """Perform PUCT search and return the best move."""
        root = PUCTNode(game.clone())

        # Perform iterations
        for _ in range(iterations):     
            node = self.selection(root)  # Select a promising node
            value = self.simulation(node)
            self.backpropagation(node, value)  # Backpropagate the results

        return root.best_child(1.0).move

    def simulation(self, node: PUCTNode):
        if not node.game.winner == None:
            return node.game.winner

        if not node.is_fully_expanded():
            node = self.expansion(node)

        # Select best child using PUCT
        best_child = node.best_child(1.0)

        # Recursively simulate
        value = self.simulation(best_child)

        return value
    

    def selection(self, node: PUCTNode) -> PUCTNode: 
        while node.game.winner == None:
            if not node.is_fully_expanded():
                return self.expansion(node)

            # best child
            node = node.best_child(1.0)
        
        return node


    def expansion(self, node: PUCTNode) -> PUCTNode:
        encoded_state = torch.tensor(node.game.encode(), dtype=torch.float32).unsqueeze(0)
        encoded_state_mtx = encoded_state.view(-1, 3, SIZE, SIZE)  # Reshape to (batch_size, 3, 9, 9)

        P_pred, Q_pred = self.network(encoded_state_mtx)

        untried_moves = [move for move in node.game.legal_moves() if move not in [child.move for child in node.children]]
        policy_matrix = P_pred.view(-1, 1, SIZE, SIZE)

        node.N += 1
        node.total_value += Q_pred
        node.Q = node.total_value / node.N

        # TODO: normelize 
        for move in untried_moves:
            x,y = move
            next_state = node.game.clone()
            next_state.make_move(move)
            prior = float(policy_matrix[0,0,x,y])
            child_node = PUCTNode(next_state, parent=node, move=move, prior=prior) 
            node.children.append(child_node)  # Add the new child node to the tree

        return node 
    
    
    def backpropagation(self, node: PUCTNode, value: float):
        while node is not None:
            node.N += 1 
            node.total_value += value
            node = node.parent  



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
    game_network: GameNetwork = GameNetwork(size)
    game_network.load()
    puct_player = PUCTPlayer(network=game_network)  # Create an PUCT player
    mcts_player = MCTSPlayer()  # Create an MCTS player

    # game_gui = GomokuGUI()
    # threading.Thread(target=game_gui.start, daemon=False).start()

    while game.winner == None:  # Play until the game ends
        # mcts_move, _ = mcts_player.choose_move(game)
        row, col = tuple(map(int, input("Your move (WHITE, enter row, col): ").split()))   
        mcts_move = (row, col) 
        if mcts_move not in game.legal_moves():
            print("Illegal move. Try again.")
            continue
        print("MCTS (Black) selected move:", mcts_move)
        x,y = mcts_move
        # game_gui.add_piece(x, y, PLAYER_BLACK)   
        game.make_move(mcts_move)
        print(game)

        if game.winner:
            break

        puct_move = puct_player.choose_move(game)
        print("PUCT (White) selected move:", puct_move)
        x,y = puct_move
        # game_gui.add_piece(x, y, PLAYER_WHITE)   
        game.make_move(puct_move)
        print(game)


    
    # game_gui.mark_winner(x, y, game.winner)

    if game.winner == PLAYER_BLACK:
        print("BLACK (MCTS) is the WINNER!")
    elif game.winner == PLAYER_WHITE:
        print("WHITE (Human) is the WINNER!")
    else:
        print("It's a draw!")  
    
 

if __name__ == "__main__": 
    play_with_puct()



     








 