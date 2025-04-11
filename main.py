'''
Made with a little bit of help from Claude 
'''
import numpy as np
import pygame
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from dqn import DQN


class KaboomEnv:
    def __init__(self, width=800, height=600, basket_width=100):
        self.width = width
        self.height = height
        self.basket_width = basket_width
        self.ball_radius = 10
        self.ball_speed = 5
        self.basket_speed = 20
        
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Kaboom!")
        self.clock = pygame.time.Clock()
        
        self.reset()

    def reset(self):
        self.basket_pos = self.width // 2
        self.ball_pos = np.random.randint(0, self.width)
        self.ball_height = 0
        self.misses = 0
        self.score = 0
        return self._get_state()

    def _get_state(self):
        return np.array([
            self.basket_pos / self.width,
            self.ball_pos / self.width,
            self.ball_height / self.height,
            (self.ball_pos - self.basket_pos) / self.width  # Relative position
        ])

    def step(self, action):
        if action == 0:  # Move left
            self.basket_pos = max(self.basket_width // 2, self.basket_pos - self.basket_speed)
        elif action == 2:  # Move right
            self.basket_pos = min(self.width - self.basket_width // 2, self.basket_pos + self.basket_speed)

        self.ball_height += self.ball_speed

        if self.ball_height >= self.height - self.ball_radius:
            if abs(self.ball_pos - self.basket_pos) < self.basket_width // 2:
                reward = 10  # Increased reward for catching
                self.score += 1
            else:
                reward = -5  # Increased penalty for missing
                self.misses += 1

            self.ball_pos = np.random.randint(0, self.width)
            self.ball_height = 0
        else:
            reward = -0.1  # Small negative reward for each step to encourage faster catching

        done = self.misses >= 3
        info = {'score': self.score}

        return self._get_state(), reward, done, info

    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.screen.fill((0, 0, 0))  # Black background
        
        # Draw the ball
        pygame.draw.circle(self.screen, (255, 0, 0), (int(self.ball_pos), int(self.ball_height)), self.ball_radius)
        
        # Draw the basket
        pygame.draw.rect(self.screen, (255, 255, 255), 
                         (int(self.basket_pos - self.basket_width // 2), self.height - 20, self.basket_width, 20))
        
        # Display misses and score
        font = pygame.font.Font(None, 36)
        misses_text = font.render(f"Misses: {self.misses}", True, (255, 255, 255))
        score_text = font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(misses_text, (10, 10))
        self.screen.blit(score_text, (10, 50))
        
        pygame.display.flip()

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

class RuleBasedAgent:
    def __init__(self, basket_speed, basket_width, width):
        self.basket_speed = basket_speed
        self.basket_width = basket_width
        self.width = width

    def get_action(self, state):
        basket_pos, ball_pos, _, _ = state
        
        # Convert normalized positions back to pixel coordinates
        basket_x = basket_pos * self.width
        ball_x = ball_pos * self.width
        
        # Calculate the number of steps needed
        steps = int((ball_x - basket_x) / self.basket_speed)
        
        # Account for the basket's width
        if abs(steps) * self.basket_speed <= self.basket_width / 2:
            return 1  # Stay in place if the ball will land within the basket's width
        elif steps > 0:
            return 2  # Move right
        else:
            return 0  # Move left

def collect_data_with_rule_based_agent(env, agent, num_episodes):
    rule_based_agent = RuleBasedAgent(basket_speed=env.basket_speed, 
                                      basket_width=env.basket_width, 
                                      width=env.width)
    total_score = 0
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action = rule_based_agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            agent.replay_buffer.append((state, action, reward, next_state, done))
            state = next_state
        total_score += info['score']
        print(f"Rule-based Agent - Episode {episode + 1}, Score: {info['score']}")
    
    average_score = total_score / num_episodes
    print(f"Rule-based Agent - Average Score over {num_episodes} episodes: {average_score:.2f}")

def train_drl_agent(env, agent, num_episodes, render=True, fps=60):
    total_score = 0
    clock = pygame.time.Clock()
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            if render:
                env.render()
                clock.tick(fps)

            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            agent.replay_buffer.append((state, action, reward, next_state, done))
            if len(agent.replay_buffer) >= agent.batch_size:
                agent.train_step()
            state = next_state
            episode_reward += reward
        
        total_score += info['score']
        if (episode + 1) % 10 == 0:
            print(f"DRL Agent Training - Episode {episode + 1}, Score: {info['score']}, "
                  f"Reward: {episode_reward:.2f}, Epsilon: {agent.get_epsilon():.4f}")
    
    average_score = total_score / num_episodes
    print(f"DRL Agent Training - Average Score over {num_episodes} episodes: {average_score:.2f}")

def play_drl_agent_visible(env, agent):
    state = env.reset()
    done = False
    while not done:
        env.render()
        action = agent.get_action(state, is_training=False)
        next_state, _, done, info = env.step(action)
        state = next_state
        env.clock.tick(60)

    print(f"DRL Agent Visible Play - Final Score: {info['score']}")

def main():
    env = KaboomEnv()
    agent = LearningAgent(input_size=4, output_size=3)

    print("Collecting data with rule-based agent...")
    collect_data_with_rule_based_agent(env, agent, num_episodes=100)

    print("\nTraining DRL agent...")
    train_drl_agent(env, agent, num_episodes=500, render=True, fps=60)

    print("\nPlaying with trained DRL agent (visible, no exploration)...")
    agent.epsilon = 0  # Set epsilon to 0 for no exploration
    while True:
        play_drl_agent_visible(env, agent)
        play_again = input("Play again? (y/n): ").lower()
        if play_again != 'y':
            break

    pygame.quit()

if __name__ == "__main__":
    main()
