import random
import numpy as np
import threading
from config import SIZE,EMPTY, STRIKE, PLAYER_BLACK, PLAYER_WHITE, DRAW
from game_gui import GomokuGUI

class GomokuGame:
    """ Gomoku game - Black (1) vs White (-1) """
 
    def __init__(self, size=9, strike=5):
        """Initialize the game"""
        self.board = [[EMPTY for _ in range(size)] for _ in range(size)]  # Create an empty board
        self.turn = PLAYER_BLACK  # Black starts first
        self.winner = None  # No winner at the beginning
        self.history = []  # Keeps track of move history for undo functionality

    def switch_turn(self):
        """Switch turn between players"""
        self.turn = PLAYER_BLACK if self.turn == PLAYER_WHITE else PLAYER_WHITE

    def make_move(self, move: tuple):
        """Apply a move to the game board"""
        x, y = move
        if self.board[x][y] != EMPTY or self.winner is not None:
            # print(f"[DEBUG] Invalid move at ({x}, {y})")  # debug print invalid move
            return  # Invalid move (position occupied or game already won)

        # print(f"[DEBUG] Player {self.turn} moved to ({x}, {y})") # debug print player move
        self.board[x][y] = self.turn  # Place the piece on the board
        # print(f"[DEBUG] Board\n{self.__str__()}")  # debug print board
        self.history.append(move)  # Store the move in history
        self.check_winner(x, y)  # Check if this move results in a win
        if self.winner is None:  # Only switch turns if no one has won
            self.switch_turn()
        # else:
        #     print(f"[DEBUG] Winner detected: {self.winner}")  # debug print winner

    def unmake_move(self, move: tuple):
        """Undo the last move"""
        if not self.history:
            return  # No moves to undo

        x, y = self.history.pop()  # Get the last move from history
        self.board[x][y] = EMPTY  # Remove the piece
        self.switch_turn()  # Switch back the turn

    def check_winner(self, x, y):
        """Check if the last move resulted in a win"""
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # Horizontal, Vertical, Diagonal (\ and /)
        player = self.board[x][y]

        # If no winner and board is full, declare a draw
        if not self.legal_moves():
            self.winner = 0  # Draw

        for dx, dy in directions:
            count = 1  # Count the current piece
            # Check forward direction
            for i in range(1, STRIKE):
                nx, ny = x + dx * i, y + dy * i
                if 0 <= nx < SIZE and 0 <= ny < SIZE and self.board[nx][ny] == player:
                    count += 1
                else:
                    break

            # Check backward direction
            for i in range(1, STRIKE):
                nx, ny = x - dx * i, y - dy * i
                if 0 <= nx < SIZE and 0 <= ny < SIZE and self.board[nx][ny] == player:
                    count += 1
                else:
                    break

            if count >= STRIKE:  # If the required number of pieces are in a row
                self.winner = player
                return
    

    def clone(self):
        """Create a deep copy of the current game state"""
        new_game = GomokuGame(SIZE, STRIKE)
        new_game.board = [row[:] for row in self.board]  # Copy the board state
        new_game.turn = self.turn
        new_game.history = self.history[:]
        return new_game
    

    def encode(self):
        """
        Encode the current game state as a tensor suitable for CNN input.

        Returns:
        - A PyTorch tensor of shape (3, board_size, board_size), where:
        - Channel 1: Player 1's pieces (1 where Player 1 has a stone, 0 elsewhere).
        - Channel 2: Player 2's pieces (1 where Player 2 has a stone, 0 elsewhere).
        - Channel 3: Current player indicator (1 if Player 1's turn, -1 if Player 2's turn).
        """
        board_size = SIZE  # Assume a square board

        # Create empty matrices for player 1 and player 2
        player1_matrix = np.zeros((board_size, board_size), dtype=np.float32)
        player2_matrix = np.zeros((board_size, board_size), dtype=np.float32)

        # Fill the matrices based on board state
        for r in range(board_size):
            for c in range(board_size):
                if self.board[r][c] == 1:
                    player1_matrix[r][c] = 1  # Player 1's stones
                elif self.board[r][c] == -1:
                    player2_matrix[r][c] = 1  # Player 2's stones

        # Create the "current player" channel
        current_player_channel = np.full((board_size, board_size), 1 if self.turn == 1 else -1, dtype=np.float32)

        # Stack the three matrices into a single tensor with shape (3, board_size, board_size)
        encoded_state = np.stack([player1_matrix, player2_matrix, current_player_channel], axis=0)

        return encoded_state.flatten()


    def decode(self, action):
        """Translate an action index into a move"""
        x = action // SIZE
        y = action % SIZE
        return x, y

    def legal_moves(self):
        """Return a list of available legal moves"""
        legal_moves = [(i, j) for i in range(SIZE) for j in range(SIZE) if self.board[i][j] == EMPTY]
        random.shuffle(legal_moves)
        return legal_moves

    def status(self):
        """Return the current game status"""
        if self.winner == PLAYER_BLACK:
            return "Black Wins"
        elif self.winner == PLAYER_WHITE:
            return "White Wins"
        elif self.winner == 0:
            return "Draw"
        return "Ongoing"

    def __str__(self):
        """Return a string representation of the game board"""
        board_str = "\n".join(" ".join(f"{cell:2}" for cell in row) for row in self.board)
        status_str = f"Status: {self.status()}\nTurn: {'Black' if self.turn == 1 else 'White'}"
        return f"{board_str}\n{status_str}"
    

def test_gomoku():
    """test gomoku game"""
    size = 9
    strike = 5

    # ✅ Test horizontal win on every row
    for row in range(size):
        game = GomokuGame(size, strike)
        for i in range(strike):
            game.make_move((row, i))  # Black places pieces
            if i < strike - 1:  # White plays in between to maintain turn order
                game.make_move((row + 1 if row + 1 < size else 0, i))
        assert game.winner == 1, f"Error: Black should win horizontally on row {row}"

    # ✅ Test vertical win on every column
    for col in range(size):
        game = GomokuGame(size, strike)
        for i in range(strike):
            game.make_move((i, col))  # Black places pieces
            if i < strike - 1:  # White plays in between
                game.make_move((i, col + 1 if col + 1 < size else 0))
        assert game.winner == 1, f"Error: Black should win vertically on column {col}"

    # ✅ Test diagonal win (\) from different starting positions
    for start in range(size - strike + 1):
        game = GomokuGame(size, strike)
        for i in range(strike):
            game.make_move((start + i, start + i))  # Black places pieces diagonally
            if i < strike - 1:  # White plays in between
                game.make_move((start + i, start + i + 1 if start + i + 1 < size else 0))
        assert game.winner == 1, f"Error: Black should win diagonally (\) from ({start},{start})"

    # ✅ Test diagonal win (/) from different starting positions
    for start in range(size - strike + 1):
        game = GomokuGame(size, strike)
        for i in range(strike):
            game.make_move((start + i, size - 1 - (start + i)))  # Black places pieces diagonally
            if i < strike - 1:  # White plays in between
                game.make_move((start + i, size - 2 - (start + i) if size - 2 - (start + i) >= 0 else 0))
        assert game.winner == 1, f"Error: Black should win diagonally (/) from ({start},{size - 1 - start})"

    # ✅ Test invalid move (placing a piece on an occupied space)
    game = GomokuGame(size, strike)
    game.make_move((0, 0))  # Black plays
    game.make_move((0, 0))  # White tries to play on the same spot
    assert game.board[0][0] == 1, "Error: Invalid move should not be allowed"
    
    # ✅ Test draw detection (no winner, board full)
    game = GomokuGame(3, 3)  # Use a small board to test quickly
    moves = [
        (0, 0), (0, 1), (0, 2),
        (1, 1), (1, 2), (1, 0),
        (2, 0), (2, 2), (2, 1)
    ]  # This pattern ensures no winning row/column/diagonal

    for i, move in enumerate(moves):
        game.make_move(move)

    # print(f"[DEBUG] Legal moves left: {game.legal_moves()}")  # Should be []
    # print(f"[DEBUG] Winner: {game.winner}")  # Should be 0 (draw)
    assert game.winner == 0, "Error: The game should end in a draw"

    # ✅ Test undo move (unmake_move)
    game = GomokuGame(size, strike)
    game.make_move((1, 1))  # Black plays
    game.make_move((2, 2))  # White plays
    game.unmake_move((2, 2))  # Undo White's move
    assert game.board[2][2] == 0, "Error: Undo move did not remove last move"
    assert game.turn == -1, "Error: Undo move did not return turn to previous player"

    # ✅ Test game cloning (clone method)
    game = GomokuGame(size, strike)
    game.make_move((3, 3))  # Black plays
    clone_game = game.clone()  # Create a copy of the game state
    assert clone_game.board == game.board, "Error: Cloned board does not match original"
    assert clone_game.turn == game.turn, "Error: Cloned game turn does not match original"

    # ✅ Test legal moves function (should not include occupied cells)
    game = GomokuGame(3, 3)
    game.make_move((1, 1))  # Black plays
    legal_moves = game.legal_moves()
    assert (1, 1) not in legal_moves, "Error: legal_moves() includes occupied cell"
    assert len(legal_moves) == 8, "Error: legal_moves() does not return correct number of moves"

    print("✅ All tests passed successfully!")


if __name__ == "__main__":
    game = GomokuGame(size=9, strike=5)  # Initialize a 9x9 Gomoku game (win with 5 in a row)
    game_gui = GomokuGUI()
    threading.Thread(target=game_gui.start, daemon=False).start() 
    # Run all tests
    #test_gomoku()
    
    DEBUG = True

    while game.winner is None:
        moves = game.legal_moves()
        if not moves:
            break  # Stop if no more moves are available
        row = int(input("Your move (WHITE, enter row: "))  # Human input
        col = int(input("Your move (WHITE, enter column: "))  # Human input
        move = (row, col) # Convert to move format
        if move not in game.legal_moves():
            print("Illegal move. Try again.")
            continue
        else:
            game_gui.add_piece(row, col, game.turn)
            game.make_move(move)  # Play a random legal move
            game.encode() # test encode
            x,y = move
        # game.make_move(move)  # Play a random legal move

    game_gui.mark_winner(x, y, game.winner)

    if DEBUG:
        print("\n[DEBUG] Game Over!")
        if game.winner:
            print(f"[DEBUG] Winner: {'Black' if game.winner == 1 else 'White'}")
        else:
            print("[DEBUG] It's a draw!")


    # print("\nFinal Board:")
    # print(game)
