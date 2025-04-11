import numpy as np
import pygame
import sys

from agent import LearningAgent

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
