import torch
import numpy as np
from e_nira_env import ENiraEnv, MOVE_ACTION_INDEX
from dqn_agent import DQNAgent # DQNAgent will now use DuelingDQN internally and PER

EPISODES = 200000 # Increased episodes significantly again
MAX_STEPS = 200
SAVE_PATH = "dqn_enira_smarter_agent_per.pth" # New save path for PER version

env = ENiraEnv()
# State size: 9 board positions + 1 for turn + 1 for phase = 11
state_size = 9 + 1 + 1 
action_size = 9 + len(MOVE_ACTION_INDEX)

agent = DQNAgent(state_size, action_size,
                 use_double_dqn=True,
                 use_prioritized_replay=True, # <<<<< ENABLE PER HERE >>>>>
                 gamma=0.995,
                 lr=5e-5, # Slightly lower learning rate for stability with PER
                 batch_size=256,
                 memory_size=100000, # Max capacity for SumTree
                 epsilon_start=1.0,
                 epsilon_end=0.005,
                 epsilon_decay=0.9998,
                 target_update=200)

print(f"Starting training for {EPISODES} episodes...")
print(f"Using Double DQN: {agent.use_double_dqn}, Prioritized Replay: {agent.use_prioritized_replay}")
print(f"State Size: {state_size}, Action Size: {action_size}")

for episode in range(EPISODES):
    state = env.reset()
    total_reward = 0
    done = False
    
    state_for_agent = state 
    
    for step in range(MAX_STEPS):
        valid_actions = env.get_valid_actions()
        
        if not valid_actions: # Check for no valid moves for the current player
            # If the current player (who is about to move) has no valid moves, it's typically a loss for them
            # or a draw. Assign a penalty and end the episode.
            reward = -5 # Penalty for getting stuck, essentially a loss for the current player
            done = True
            total_reward += reward
            # The agent might not even select an action if valid_actions is empty.
            # This 'if not valid_actions' block ensures the episode terminates gracefully.
            break 
            
        action = agent.select_action(state_for_agent, valid_actions)
        next_state, reward, done, _ = env.step(action)

        agent.store(state_for_agent, action, reward, next_state, done)
        agent.update()

        state_for_agent = next_state
        total_reward += reward
        if done:
            break

    if (episode + 1) % 1000 == 0: # Print less frequently for long training
        print(f"Episode {episode+1}: Reward={total_reward:.2f}, Epsilon={agent.epsilon:.4f}, Memory Size: {len(agent.memory.data) if agent.use_prioritized_replay else len(agent.memory)}")

torch.save(agent.policy_net.state_dict(), SAVE_PATH)
print(f"Model saved to {SAVE_PATH}")