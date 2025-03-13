from config import BLACK_COLOR, WHITE_COLOR, BROWN_COLOR, SIZE, WIDTH, HEIGHT, CELL_SIZE, PLAYER_BLACK, PLAYER_WHITE, STRIKE
import pygame
 

# Define a custom event for drawing a circle
DRAW_CIRCLE_EVENT = pygame.USEREVENT + 1
DRAW_WINNER_EVENT = pygame.USEREVENT + 2 

class Piece(pygame.sprite.Sprite):
    def __init__(self, x, y, color):
        super().__init__()
        self.image = pygame.Surface((30, 30), pygame.SRCALPHA)
        pygame.draw.circle(self.image, color, (15, 15), 15)
        self.rect = self.image.get_rect(center=(y,x))

class GomokuGUI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Gomoku")
        self.clock = pygame.time.Clock()
        self.pieces = pygame.sprite.Group()
        self.board = [[None] * SIZE for _ in range(SIZE)]
        self.winning_positions = []
        self.winner = None
        self.font = pygame.font.Font(None, 50) 


    def draw_board(self):
        self.screen.fill(BROWN_COLOR)
        for i in range(SIZE+1):
            pygame.draw.line(self.screen, BLACK_COLOR, 
                            (CELL_SIZE // 2, i * CELL_SIZE + CELL_SIZE // 2), 
                            (WIDTH - CELL_SIZE // 2, i * CELL_SIZE + CELL_SIZE // 2), 2)
         
            pygame.draw.line(self.screen, BLACK_COLOR, (i * CELL_SIZE + CELL_SIZE // 2, CELL_SIZE // 2), 
                             (i * CELL_SIZE + CELL_SIZE // 2, HEIGHT - CELL_SIZE // 2), 2)

        # Draw all stored pieces
        self.pieces.draw(self.screen)
        
        if self.winner:
            color = {PLAYER_BLACK: (0, 255, 0), PLAYER_WHITE:(255, 0, 0)}.get(self.winner)
             
            # for x, y in self.winning_positions:
            #     pygame.draw.circle(self.screen, color, (x * CELL_SIZE + CELL_SIZE, y * CELL_SIZE + CELL_SIZE), 15, 4) 
            
            # Show winner message
            winner_text = "Black Wins!" if self.winner == PLAYER_BLACK else "White Wins!"
            text_surface = self.font.render(winner_text, True, color)
            text_rect = text_surface.get_rect(center=(WIDTH // 2, HEIGHT - 15))
            self.screen.blit(text_surface, text_rect)
        
        pygame.display.flip()
 
    def add_piece(self, x,y, player):
        self.board[x][y] = player
        color = {PLAYER_BLACK: BLACK_COLOR, PLAYER_WHITE: WHITE_COLOR}.get(player)
        pygame.event.post(pygame.event.Event(DRAW_CIRCLE_EVENT, {"x": x, "y": y, "color": color}))

    def mark_winner(self, x,y, winner):
        pygame.event.post(pygame.event.Event(DRAW_WINNER_EVENT, {"x": x, "y": y, "winner": winner}))

    def update_winning_positions(self, last_x, last_y, winner):
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # Right, Down, Diagonal Right-Down, Diagonal Right-Up

        for dx, dy in directions:
            count = 1
            winning_positions = [(last_y, last_x)]  # Start with the last move

            # Check backwards in this direction
            for i in range(1, STRIKE):
                nx, ny = last_x - i * dx, last_y - i * dy
                if 0 <= nx < SIZE and 0 <= ny < SIZE and self.board[nx][ny] == winner:
                    count += 1
                    winning_positions.append((nx, ny))
                else:
                    break  # Stop if a piece doesn't match

            # Check forwards in this direction
            for i in range(1, STRIKE):
                nx, ny = last_x + i * dx, last_y + i * dy
                if 0 <= nx < SIZE and 0 <= ny < SIZE and self.board[nx][ny] == winner:
                    count += 1
                    winning_positions.append((nx, ny))
                else:
                    break  # Stop if a piece doesn't match

            # If we found STRIKE or more in a row, store the positions and return the winner
            if count >= STRIKE:
                self.winning_positions = winning_positions
               

    def handle_events(self):
        """ Handle user inputs and custom events """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit()
                return False
            elif event.type == DRAW_CIRCLE_EVENT:
                piece = Piece(event.x * CELL_SIZE + CELL_SIZE, event.y * CELL_SIZE + CELL_SIZE, event.color)
                self.pieces.add(piece) 
            elif event.type == DRAW_WINNER_EVENT:
                self.winner = event.winner
                self.update_winning_positions(event.x, event.y, event.winner)

        return True


    def start(self):
        """ Starts the game loop """
        while self.handle_events():
            self.draw_board()
            self.clock.tick(30)

# # Run the game
# if __name__ == "__main__":
#     game = GomokuGUI()
#     # game.start()

#     # Start the game in a separate thread
#     import threading
#     threading.Thread(target=game.start, daemon=False).start()    

