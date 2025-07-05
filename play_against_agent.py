import pygame
import sys
import torch
import numpy as np
from e_nira_env import ENiraEnv, VALID_MOVES, MOVE_ACTION_INDEX, INDEX_TO_MOVE_ACTION
from dqn_agent import DuelingDQN # Import DuelingDQN directly

# --- Game Display Constants ---
WIDTH, HEIGHT = 600, 600
WHITE = (255, 255, 255)
ORANGE = (255, 165, 0)
BLUE = (0, 0, 255)
BOARD_COLOR = (0, 0, 0)
NODE_RADIUS = 20
FONT_SIZE = 36
LINE_WIDTH = 5

NODES = {
    1: (100, 100), 2: (300, 100), 3: (500, 100),
    4: (100, 300), 5: (300, 300), 6: (500, 300),
    7: (100, 500), 8: (300, 500), 9: (500, 500)
}

# --- Load Trained DQN Model ---
MODEL_PATH = "dqn_enira_smarter_agent_per.pth" # <<< UPDATED MODEL PATH >>>
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Adjust state_size to match the new environment state representation (9 board + 1 turn + 1 phase)
policy_net = DuelingDQN(9 + 1 + 1, 9 + len(MOVE_ACTION_INDEX)).to(device) # Use DuelingDQN
policy_net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
policy_net.eval()

# --- AI Action Selection ---
def get_ai_action(state, valid_actions):
    # Ensure state is a numpy array before converting to tensor
    state_np = np.array(state, dtype=np.float32)
    state_tensor = torch.from_numpy(state_np).unsqueeze(0).to(device)
    
    with torch.no_grad():
        q_values = policy_net(state_tensor).cpu().numpy().flatten()
    
    valid_q_values = {a: q_values[a] for a in valid_actions}
    if not valid_q_values:
        # Fallback if no valid actions are available (should indicate game end or error)
        return random.choice(range(9 + len(MOVE_ACTION_INDEX)))
    
    best_action = max(valid_q_values, key=valid_q_values.get)
    return best_action

# --- Pygame Setup ---
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Play E-Nira vs Trained DQN Agent")
font = pygame.font.Font(None, FONT_SIZE)
clock = pygame.time.Clock()

def draw_board(state, winner=None):
    screen.fill(WHITE)
    for start, ends in VALID_MOVES.items():
        for end in ends:
            pygame.draw.line(screen, BOARD_COLOR, NODES[start], NODES[end], LINE_WIDTH)

    # Only draw the board part of the state for visual representation
    board_state_only = state[:9] 
    for i, value in enumerate(board_state_only):
        if value == 1:
            pygame.draw.circle(screen, ORANGE, NODES[i + 1], NODE_RADIUS)
        elif value == -1:
            pygame.draw.circle(screen, BLUE, NODES[i + 1], NODE_RADIUS)

    if winner is not None:
        text = f"{'Orange' if winner == 1 else 'Blue'} Wins!"
        if winner == "Draw":
            text = "Draw!"
            color = (128, 128, 128) # Grey for draw
        else:
            color = ORANGE if winner == 1 else BLUE
        render = font.render(text, True, color)
        screen.blit(render, (WIDTH // 2 - render.get_width() // 2, 20))

    pygame.display.flip()

# --- Game Logic ---
env = ENiraEnv()
state = env.reset() # This now returns the augmented state
winner = None
selected_piece = None

running = True
while running:
    # Pass only the board part to draw_board for visualization
    draw_board(state, winner) 

    if env.turn == -1 and not winner:
        pygame.time.wait(300)
        valid_actions = env.get_valid_actions()
        
        if not valid_actions:
            # If AI has no valid moves, it loses
            winner = 1 # Human wins
            env.steps = env.MAX_STEPS # Simulate max steps reached to trigger end
            print("AI has no valid moves. Human wins!")
            continue 

        action = get_ai_action(state, valid_actions) # Pass full state to AI
        next_state, reward, done, _ = env.step(action)
        state = next_state
        if done:
            if reward > 0:
                winner = -1 # AI wins
            elif reward < 0:
                winner = 1 # AI loses (human wins)
            else: # reward is 0, typically for max steps reached or draw condition
                winner = "Draw" 
            print(f"Game over. Winner: {winner}")

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN and env.turn == 1 and not winner:
            pos = pygame.mouse.get_pos()
            human_action = None 
            
            for node, coord in NODES.items():
                x, y = coord
                if (x - NODE_RADIUS < pos[0] < x + NODE_RADIUS and
                        y - NODE_RADIUS < pos[1] < y + NODE_RADIUS):
                    node_idx = node - 1

                    if env.phase == "placing":
                        if state[node_idx] == 0: # Ensure node is empty
                            human_action = node_idx
                            break
                    elif env.phase == "moving":
                        if selected_piece is None:
                            # Check if the clicked piece belongs to the human (player 1)
                            if state[node_idx] == 1: 
                                selected_piece = node_idx
                            break 
                        else:
                            to_idx = node_idx
                            # Check if destination is empty and it's a valid move from selected_piece
                            if state[to_idx] == 0 and (to_idx + 1) in VALID_MOVES[selected_piece + 1]:
                                human_action = 9 + MOVE_ACTION_INDEX[(selected_piece, to_idx)]
                                selected_piece = None # Reset selected piece after a valid move
                                break
                            else: # Invalid move or selection for the second click
                                selected_piece = None 
                                break 

            if human_action is not None:
                current_valid_human_actions = env.get_valid_actions()
                if human_action in current_valid_human_actions:
                    next_state, reward, done, _ = env.step(human_action)
                    state = next_state
                    if done:
                        if reward > 0:
                            winner = 1 # Human wins
                        elif reward < 0:
                            winner = -1 # Human loses (AI wins)
                        else:
                            winner = "Draw"
                        print(f"Game over. Winner: {winner}")
                else:
                    print("Invalid human move attempted!")
            
    clock.tick(30)

pygame.quit()
sys.exit()