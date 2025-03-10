import pygame
import sys

# Constants
WIDTH, HEIGHT = 600, 600
BOARD_COLOR = (0, 0, 0)
WHITE = (255, 255, 255)
ORANGE = (255, 165, 0)
BLUE = (0, 0, 255)
NODE_RADIUS = 20
LINE_WIDTH = 5
FONT_SIZE = 36
ANIMATION_SPEED = 20

# Node positions
NODES = {
    1: (100, 100), 2: (300, 100), 3: (500, 100),
    4: (100, 300), 5: (300, 300), 6: (500, 300),
    7: (100, 500), 8: (300, 500), 9: (500, 500)
}

# Winning combinations
WINNING_COMBINATIONS = [
    {1, 2, 3}, {4, 5, 6}, {7, 8, 9},  # Rows
    {1, 4, 7}, {2, 5, 8}, {3, 6, 9},  # Columns
    {1, 5, 9}, {3, 5, 7}  # Diagonals
]

# Valid moves
VALID_MOVES = {
    1: [2, 4, 5], 2: [1, 3, 5], 3: [2, 5, 6],
    4: [1, 5, 7], 5: [1, 2, 3, 4, 6, 7, 8, 9], 6: [3, 5, 9],
    7: [4, 5, 8], 8: [5, 7, 9], 9: [5, 6, 8]
}

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("E-Nira Game")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, FONT_SIZE)
        self.turn = ORANGE
        self.pieces = {ORANGE: [None] * 3, BLUE: [None] * 3}
        self.placed_pieces = {ORANGE: [], BLUE: []}
        self.selected_piece = None
        self.running = True
        self.phase = "placing"
        self.winner = None
        self.moving_piece = None
        self.moving_target = None
        self.moving_position = None
        self.show_winner_message = False

    def draw_board(self):
        self.screen.fill(WHITE)
        for start, ends in VALID_MOVES.items():
            for end in ends:
                pygame.draw.line(self.screen, BOARD_COLOR, NODES[start], NODES[end], LINE_WIDTH)
        for node, pos in NODES.items():
            pygame.draw.circle(self.screen, BOARD_COLOR, pos, NODE_RADIUS, 3)
        self.draw_pieces()
        self.draw_turn()
        if self.show_winner_message:
            self.display_winner_text()

    def draw_pieces(self):
        for color, pieces in self.placed_pieces.items():
            for node in pieces:
                if node != self.moving_piece:
                    pos = NODES[node]
                    pygame.draw.circle(self.screen, color, pos, NODE_RADIUS)
        if self.moving_piece:
            pygame.draw.circle(self.screen, self.turn, self.moving_position, NODE_RADIUS)

    def draw_turn(self):
        if self.winner:
            return
        text = f"{('Orange' if self.turn == ORANGE else 'Blue')} to move"
        text_color = ORANGE if self.turn == ORANGE else BLUE
        render = self.font.render(text, True, text_color)
        self.screen.blit(render, (WIDTH // 2 - render.get_width() // 2, 20))

    def check_winner(self):
        for color in [ORANGE, BLUE]:
            if len(self.placed_pieces[color]) == 3:
                nodes = set(self.placed_pieces[color])
                for combination in WINNING_COMBINATIONS:
                    if combination.issubset(nodes):
                        self.winner = color
                        self.show_winner_message = True
                        return color
        return None

    def animate_move(self):
        if self.moving_piece and self.moving_position:
            current_x, current_y = self.moving_position
            target_x, target_y = NODES[self.moving_target]
            if abs(current_x - target_x) < ANIMATION_SPEED and abs(current_y - target_y) < ANIMATION_SPEED:
                self.placed_pieces[self.turn].remove(self.moving_piece)
                self.placed_pieces[self.turn].append(self.moving_target)
                self.moving_piece = None
                self.turn = BLUE if self.turn == ORANGE else ORANGE
                self.check_winner()
            else:
                dx = (target_x - current_x) / ANIMATION_SPEED
                dy = (target_y - current_y) / ANIMATION_SPEED
                self.moving_position = (current_x + dx, current_y + dy)

    def display_winner_text(self):
        text = f"{('Orange' if self.winner == ORANGE else 'Blue')} wins! Play again? (Y/N)"
        text_color = ORANGE if self.winner == ORANGE else BLUE
        render = self.font.render(text, True, self.winner)
        self.screen.blit(render, (WIDTH // 2 - render.get_width() // 2, 40))

    def restart_game(self):
        self.__init__()

    def handle_click(self, pos):
        if self.winner or self.moving_piece:
            return
        for node, position in NODES.items():
            if (position[0] - NODE_RADIUS < pos[0] < position[0] + NODE_RADIUS and
                    position[1] - NODE_RADIUS < pos[1] < position[1] + NODE_RADIUS):
                if self.phase == "placing" and node not in self.placed_pieces[ORANGE] and node not in self.placed_pieces[BLUE]:
                    if None in self.pieces[self.turn]:
                        self.pieces[self.turn][self.pieces[self.turn].index(None)] = node
                        self.placed_pieces[self.turn].append(node)
                        self.turn = BLUE if self.turn == ORANGE else ORANGE
                        if all(None not in self.pieces[c] for c in [ORANGE, BLUE]):
                            self.phase = "moving"
                            self.check_winner()
                elif self.phase == "moving":
                    if self.selected_piece is None:
                        if node in self.placed_pieces[self.turn]:
                            self.selected_piece = node
                    else:
                        if node in VALID_MOVES[self.selected_piece] and node not in self.placed_pieces[ORANGE] and node not in self.placed_pieces[BLUE]:
                            self.moving_piece = self.selected_piece
                            self.moving_target = node
                            self.moving_position = NODES[self.selected_piece]
                            self.selected_piece = None

    def run(self):
        while self.running:
            self.draw_board()
            self.animate_move()
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(event.pos)
                elif event.type == pygame.KEYDOWN and self.winner:
                    if event.key == pygame.K_y:
                        self.restart_game()
                    elif event.key == pygame.K_n:
                        self.running = False
            self.clock.tick(30)
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    Game().run()
