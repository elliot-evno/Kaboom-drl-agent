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
        self.temperature = 5
        self.train_frequency = 4
        self.target_update_frequency = 1000
        self.total_steps = 0

    def get_action(self, state, is_training=True):
        with torch.no_grad():
            q_values = self.dqn(torch.FloatTensor(state))
            q_values = torch.softmax(q_values/self.temperature, dim=0)
            return  torch.multinomial(q_values, num_samples=1, replacement=True)

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.cat([s.unsqueeze(0) for s in states])
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.cat([s.unsqueeze(0) for s in next_states])
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

        if self.total_steps % 300 == 0:
            if self.temperature > 1:
                self.temperature = max(1, self.temperature * 0.99)
                print(f"Temperature: {self.temperature}")
            else:
                self.temperature = min(1, self.temperature * 1.01)
                print(f"Temperature: {self.temperature}")

    def get_temperature(self):
        return self.temperature