import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from dqn import DQN

class LearningAgent:
    def __init__(self, input_size, output_size):
        self.dqn = DQN(input_size, output_size)
        self.target_dqn = DQN(input_size, output_size)
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=0.0001)
        self.criterion = nn.MSELoss()
        
        self.replay_buffer = deque(maxlen=100000)
        self.batch_size = 128
        self.gamma = 0.99
        
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay_steps = 100000  # Number of steps over which to decay epsilon
        self.epsilon = self.epsilon_start
        
        self.train_frequency = 4
        self.target_update_frequency = 1000
        self.total_steps = 0

    def get_action(self, state, is_training=True):
        if is_training and random.random() < self.epsilon:
            return random.randint(0, 2)
        else:
            with torch.no_grad():
                q_values = self.dqn(torch.FloatTensor(state))
                return q_values.argmax().item()

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        current_q_values = self.dqn(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_dqn(next_states).max(1)[0]
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        loss = self.criterion(current_q_values, target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.total_steps += 1
        if self.total_steps % self.target_update_frequency == 0:
            self.target_dqn.load_state_dict(self.dqn.state_dict())

        # Update epsilon based on total steps
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start - (self.epsilon_start - self.epsilon_end) * (self.total_steps / self.epsilon_decay_steps)
        )

    def get_epsilon(self):
        return self.epsilon