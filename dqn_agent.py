import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from sumtree import SumTree # Import SumTree

# DuelingDQN from previous iteration
class DuelingDQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DuelingDQN, self).__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size)
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        features = self.feature_layer(x)
        advantage = self.advantage_stream(features)
        value = self.value_stream(features)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.99, lr=1e-3,
                 batch_size=64, memory_size=10000, epsilon_start=1.0,
                 epsilon_end=0.01, epsilon_decay=0.995, target_update=10,
                 use_double_dqn=True, use_prioritized_replay=False):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.train_step = 0

        self.use_double_dqn = use_double_dqn
        self.use_prioritized_replay = use_prioritized_replay

        # PER parameters
        if self.use_prioritized_replay:
            self.memory = SumTree(memory_size)
            self.alpha = 0.6  # controls how much prioritization is used (0 is uniform, 1 is full priority)
            self.beta = 0.4   # controls the importance sampling weight annealing (starts at beta, anneals to 1)
            self.beta_increment_per_sampling = 0.001
            self.max_priority = 1.0 # Initial max priority for new experiences
        else:
            self.memory = deque(maxlen=memory_size) # Regular replay buffer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DuelingDQN(state_size, action_size).to(self.device)
        self.target_net = DuelingDQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss(reduction='none') # Change to 'none' for PER (apply weights manually)

    def select_action(self, state, valid_actions):
        if np.random.rand() < self.epsilon:
            return random.choice(valid_actions)
        
        state_np = np.array(state, dtype=np.float32)
        state_tensor = torch.from_numpy(state_np).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.policy_net(state_tensor).cpu().numpy().flatten()
        
        valid_q_values = {a: q_values[a] for a in valid_actions}
        if not valid_q_values:
            return random.choice(range(self.action_size))
        
        best_action = max(valid_q_values, key=valid_q_values.get)
        return best_action

    def store(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        if self.use_prioritized_replay:
            self.memory.add(self.max_priority, experience) # Use max_priority for new samples
        else:
            self.memory.append(experience)

    def update(self):
        if self.use_prioritized_replay and self.memory.n_entries < self.batch_size:
            return
        elif not self.use_prioritized_replay and len(self.memory) < self.batch_size:
            return

        if self.use_prioritized_replay:
            # Sample from SumTree
            minibatch_indices = np.empty((self.batch_size,), dtype=np.int32)
            minibatch_priorities = np.empty((self.batch_size,), dtype=np.float32)
            minibatch_experiences = []
            
            total_p = self.memory.total_priority()
            segment = total_p / self.batch_size

            # Anneal beta for importance sampling weights
            self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)

            for i in range(self.batch_size):
                a = segment * i
                b = segment * (i + 1)
                s = random.uniform(a, b) # Sample uniformly within each segment
                
                idx, priority, data = self.memory.get(s)
                minibatch_indices[i] = idx
                minibatch_priorities[i] = priority
                minibatch_experiences.append(data)
            
            states, actions, rewards, next_states, dones = zip(*minibatch_experiences)

            # Calculate importance sampling weights
            min_prob = np.min(minibatch_priorities) / total_p
            max_weight = (min_prob * self.batch_size) ** (-self.beta)
            weights = (np.array(minibatch_priorities) / total_p * self.batch_size) ** (-self.beta)
            weights = weights / max_weight # Normalize weights
            weights_tensor = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

        else: # Standard replay
            minibatch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, next_states, dones = zip(*minibatch)
            weights_tensor = torch.ones(self.batch_size, 1).to(self.device) # All weights are 1 for uniform sampling


        # Convert to tensors
        states = torch.FloatTensor(np.array(states, dtype=np.float32)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states, dtype=np.float32)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Compute Q-values for current states
        q_values = self.policy_net(states).gather(1, actions)

        # Compute target Q-values
        with torch.no_grad():
            if self.use_double_dqn:
                next_q_values_policy = self.policy_net(next_states)
                next_actions = next_q_values_policy.max(1)[1].unsqueeze(1)
                max_next_q_values = self.target_net(next_states).gather(1, next_actions)
            else:
                max_next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        # Calculate TD-errors for PER
        td_errors = torch.abs(q_values - target_q_values).detach().cpu().numpy()
        
        # Calculate loss, applying importance sampling weights for PER
        loss = (self.loss_fn(q_values, target_q_values) * weights_tensor).mean() # Apply weights and then mean

        self.optimizer.zero_grad()
        loss.backward()
        # Optional: Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0) # Clip gradients to max norm of 1.0
        self.optimizer.step()

        # Update priorities in SumTree for PER
        if self.use_prioritized_replay:
            # Add a small epsilon (1e-5) to priorities to ensure no zero priority
            new_priorities = (td_errors + 1e-5) ** self.alpha
            for idx, p in zip(minibatch_indices, new_priorities):
                self.memory.update(idx, p)
            self.max_priority = max(self.max_priority, np.max(new_priorities)) # Update max_priority

        self.train_step += 1
        if self.train_step % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay