import os
import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
from reservoirpy.nodes import Reservoir
import pickle

#ENV_NAME = "CartPole-v1"
#ENV_NAME = "Acrobot-v1"
ENV_NAME = "MountainCar-v0"
DIRECTORY = "Matsuki_2022_data"

OBSERVATION_SPACE_AFTER_PREPROCESSING = (
    2 if ENV_NAME == "CartPole-v1" else
    4 if ENV_NAME == "Acrobot-v1" else
    1 if ENV_NAME == "MountainCar-v0" else
    None
)

# Configuration parameters for the whole setup
seed = 42
max_steps_per_episode = 200  # An episode ends when 200 steps have elapsed
max_episodes = 1  # Number of episodes to evaluate

# Use the Atari environment
env = gym.make(ENV_NAME, render_mode="human")

# To make these tasks require time-series processing, 
# the tasks are modified so that velocity and angular velocity are unavailable in the experiment.
# The observed inputs from the environment were normalized so that the range was −1 to 1
def preprocess(observation):
    if ENV_NAME == "CartPole-v1":
        # 0 : Cart Position, 1 : Cart Velocity, 2 : Pole Angle, 3 : Pole Angular Velocity
        observation_pos = observation[0] / 4.8
        observation_angle = observation[2] / 0.418
        return np.array([[observation_pos, observation_angle]])
    elif ENV_NAME == "Acrobot-v1":
        # 0 : cos(theta1), 1 : sin(theta1), 2 : cos(theta2), 3 : sin(theta2), 4 : thetaDot1, 5 : thetaDot2
        observation_cos1 = observation[0]
        observation_sin1 = observation[1]
        observation_cos2 = observation[2]
        observation_sin2 = observation[3]
        return np.array([[observation_cos1, observation_sin1, observation_cos2, observation_sin2]])
    elif ENV_NAME == "MountainCar-v0":
        # 0 : position, 1 : velocity
        # (s1t + 0.3)/0.9 
        observation_pos = (observation[0] + 0.3) / 0.9
        return np.array([[observation_pos]])

np.random.seed(seed)
torch.manual_seed(seed)

# Number of actions the agent can take
num_actions = env.action_space.n

# A multi-layered readout is a neural network having a hidden layer 
# that consists of 250 neurons whose activation function is ReLu
class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(50 + OBSERVATION_SPACE_AFTER_PREPROCESSING, 250)
        self.fc2 = nn.Linear(250, num_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Load pre-trained model
model = QNetwork()
model.load_state_dict(torch.load(f"{DIRECTORY}/{ENV_NAME}/Matsuki_2022_DQN_Final.pth", map_location=torch.device('cpu'), weights_only=True))
model.eval()

# Load the reservoir
with open(f"{DIRECTORY}/{ENV_NAME}/Matsuki_2022_Reservoir.pkl", "rb") as rs:
    reservoir = pickle.load(rs)

episode_rewards = []

for episode in range(max_episodes):
    state, _ = env.reset()
    state = preprocess(state)
    state_reservoir = reservoir(state, reset=True)
    state = torch.tensor(np.concatenate([state_reservoir, state], axis=1), dtype=torch.float32)
    episode_reward = 0
    frame = 1

    for timestep in range(1, max_steps_per_episode):
        with torch.no_grad():
            action = model(state).argmax(1).item()

        state_next, reward, done, _, _ = env.step(action)
        state_next = preprocess(state_next)
        state_next_reservoir = reservoir(state_next, reset=False)
        state_next = torch.tensor(np.concatenate([state_next_reservoir, state_next], axis=1), dtype=torch.float32)

        episode_reward += reward
        state = state_next

        frame += 1

        if done:
            break

    episode_rewards.append(episode_reward)
    print(f"Episode {episode + 1}: Steps = {frame}")

average_reward = np.mean(episode_rewards)
print(f"Average Reward over {max_episodes} episodes: {average_reward}")