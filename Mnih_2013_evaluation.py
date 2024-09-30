import os
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack
import torch
import torch.nn as nn
import numpy as np


episode_to_load = 4_000

# Configuration parameters for the whole setup
seed = 42
gamma = 0.99  # Discount factor for past rewards
epsilon = 0.0  # Epsilon greedy parameter for testing
max_steps_per_episode = 10000
max_episodes = 1  # Number of episodes to run for testing

# Use the Atari environment
env = gym.make("BreakoutNoFrameskip-v4", render_mode="human")
env = AtariPreprocessing(env)
env = FrameStack(env, 4)
env.unwrapped.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Number of actions the agent can take
num_actions = 4

class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Load pre-trained model
model = QNetwork()
model.load_state_dict(torch.load(f"DQN_model_{episode_to_load}.pth", weights_only=True))
model.eval()

episode_reward_history = []
running_reward = 0
episode_count = 0

while episode_count < max_episodes:
    observation, _ = env.reset()
    state = np.array(observation)
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    episode_reward = 0

    for timestep in range(1, max_steps_per_episode):
        # Epsilon-greedy action selection
        if epsilon > np.random.rand(1)[0]:
            action = np.random.choice(num_actions)
        else:
            with torch.no_grad():
                action = model(state).argmax(1).item()

        # Step environment
        state_next, reward, done, _, _ = env.step(action)
        state_next = np.array(state_next)
        state_next = torch.tensor(state_next, dtype=torch.float32).unsqueeze(0)

        episode_reward += reward
        state = state_next

        if done:
            break

    # Update running reward
    episode_reward_history.append(episode_reward)
    if len(episode_reward_history) > 100:
        del episode_reward_history[:1]
    running_reward = np.mean(episode_reward_history)

    episode_count += 1

    print(f"Episode {episode_count} - Reward: {episode_reward}, Running Reward: {running_reward:.2f}")

print("Testing completed.")