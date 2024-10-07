import os
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from reservoirpy.nodes import Reservoir
import pandas as pd

ENV_NAME = "BreakoutNoFrameskip-v4"
DIRECTORY = "RC_DQN_data"

# Configuration parameters for the whole setup
seed = 42
gamma = 0.99  # Discount factor for past rewards
epsilon = 1.0  # Epsilon greedy parameter
epsilon_min = 0.1  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Maximum epsilon greedy parameter
epsilon_interval = epsilon_max - epsilon_min  # Rate of decay for random actions
batch_size = 32  # Size of batch taken from replay buffer
max_steps_per_episode = 10000
max_episodes = 0  # Limit training episodes, will run until solved if smaller than 1

# Use the Atari environment
env = gym.make(ENV_NAME)
env = AtariPreprocessing(env, grayscale_obs=False)
#env = FrameStack(env, 4)
env.unwrapped.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Number of actions the agent can take
num_actions = 4

# CNN for doensampling the game frames
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=8, stride=4, padding=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128, 512)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


# DQN MLP model
class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        # forward pass with ReLU activation
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Normalize the input image
def normalize_input(image):
    """Normalize the input image to be between 0 and 1"""
    return image / 255.0


# Create DQN models
model = QNetwork()
model_target = QNetwork()
model_target.load_state_dict(model.state_dict())
model_target.eval()

# Create CNN model
cnn = CNN().eval()
#cnn = SimpleCNN().eval()
# Save the CNN model
torch.save(cnn.state_dict(), f"{DIRECTORY}/{ENV_NAME}/CNN.pth")

# Create Reservoir
reservoir = Reservoir(
                units=512,
                # low sr -> stable dynamics
                # high sr -> chaotic dynamics
                sr=0.95,
                # high lr ->  low inertia, low recall of previous states
                # low lr -> high inertia, high recall of previous states
                lr=0.9,
                #input_scaling=10.0,
                seed=seed,
            )
# Save the Reservoir model
with open(f"{DIRECTORY}/{ENV_NAME}/Reservoir.pkl", "wb") as f:
    pickle.dump(reservoir, f)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.00025)

# Replay buffer
action_history = []
state_history = []
state_next_history = []
rewards_history = []
done_history = []
episode_reward_history = []
running_reward = 0
episode_count = 0
frame_count = 0
epsilon_random_frames = 50000
epsilon_greedy_frames = 1000000.0
max_memory_length = 100000
update_after_actions = 4
update_target_network = 10000

save_model_episodes = 1000

# Loss function
loss_function = nn.SmoothL1Loss()  # Huber Loss

while True:
    observation, _ = env.reset()
    state = normalize_input(observation)
    state = np.array(state).reshape(3, 84, 84) # Add channel dimension
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    # Use CNN to downsample the game frames
    state_cnn = cnn(state).detach()
    # Use Reservoir to store the state
    state_res = reservoir(state_cnn.numpy().flatten(), reset=True)
    state_res = torch.tensor(state_res, dtype=torch.float32)
    # Concatenate the CNN and Reservoir states
    state = torch.cat((state_cnn, state_res), dim=1)

    episode_reward = 0

    for timestep in range(1, max_steps_per_episode):
        frame_count += 1

        # Epsilon-greedy action selection
        if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
            action = np.random.choice(num_actions)
        else:
            with torch.no_grad():
                action = model(state).argmax(1).item()

        epsilon -= epsilon_interval / epsilon_greedy_frames
        epsilon = max(epsilon, epsilon_min)

        # Step environment
        state_next, reward, done, _, _ = env.step(action)
        state_next = normalize_input(state_next)
        state_next = np.array(state_next).reshape(3, 84, 84) # Add channel dimension
        state_next = torch.tensor(state_next, dtype=torch.float32).unsqueeze(0) # Add batch dimension
        # Use CNN to downsample the game frames
        state_next_cnn = cnn(state_next).detach()
        # Use Reservoir to store the state
        state_next_res = reservoir(state_next_cnn.numpy().flatten(), reset=False)
        state_next_res = torch.tensor(state_next_res, dtype=torch.float32)
        # Concatenate the CNN and Reservoir states
        state_next = torch.cat((state_next_cnn, state_next_res), dim=1)

        episode_reward += reward

        # Store transition in replay buffer
        action_history.append(action)
        state_history.append(state)
        state_next_history.append(state_next)
        rewards_history.append(reward)
        done_history.append(done)

        state = state_next

        # Update the model every `update_after_actions`
        if frame_count % update_after_actions == 0 and len(done_history) > batch_size:
            # Sample from the replay buffer
            indices = np.random.choice(range(len(done_history)), size=batch_size)

            state_sample = torch.cat([state_history[i] for i in indices])
            state_next_sample = torch.cat([state_next_history[i] for i in indices])
            rewards_sample = torch.tensor([rewards_history[i] for i in indices], dtype=torch.float32)
            action_sample = torch.tensor([action_history[i] for i in indices], dtype=torch.long)
            done_sample = torch.tensor([float(done_history[i]) for i in indices], dtype=torch.float32)

            # Compute the Q-values for next states using the target network
            with torch.no_grad():
                future_rewards = model_target(state_next_sample).max(1)[0]
                updated_q_values = rewards_sample + gamma * future_rewards * (1 - done_sample)

            # Get the Q-values for the actions taken
            q_values = model(state_sample)
            q_action = q_values.gather(1, action_sample.unsqueeze(1)).squeeze(1)

            # Compute the loss
            loss = loss_function(q_action, updated_q_values)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Update the target network
        if frame_count % update_target_network == 0:
            model_target.load_state_dict(model.state_dict())
            print(f"Running reward: {running_reward:.2f} at episode {episode_count}, frame count {frame_count}, epsilon {epsilon:.2f}")

            # Save the training progress using pandas
            df = pd.DataFrame({
                    "Episode": episode_count,
                    "Reward": episode_reward,
                }, index=[0])

            if not os.path.exists(f"{DIRECTORY}/{ENV_NAME}/training_stats.csv"):
                df.to_csv(f"{DIRECTORY}/{ENV_NAME}/training_stats.csv", index=False)
            else:
                df.to_csv(f"{DIRECTORY}/{ENV_NAME}/training_stats.csv", mode="a", header=False, index=False)
            

        # Limit memory size
        if len(rewards_history) > max_memory_length:
            del rewards_history[:1]
            del state_history[:1]
            del state_next_history[:1]
            del action_history[:1]
            del done_history[:1]

        if done:
            break

    # Update running reward
    episode_reward_history.append(episode_reward)
    if len(episode_reward_history) > 100:
        del episode_reward_history[:1]
    running_reward = np.mean(episode_reward_history)

    episode_count += 1

    # Save model
    if episode_count % save_model_episodes == 0:
        torch.save(model.state_dict(), f"{DIRECTORY}/{ENV_NAME}/DQN_{episode_count}.pth")

    if running_reward > 40:
        print(f"Solved at episode {episode_count}!")
        break

    if max_episodes > 0 and episode_count >= max_episodes:
        print(f"Stopped at episode {episode_count}!")
        break
