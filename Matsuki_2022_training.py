import os
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from reservoirpy.nodes import Reservoir
import pickle
import pandas as pd

ENV_NAME = "CartPole-v1"
#ENV_NAME = "Acrobot-v1"
#ENV_NAME = "MountainCar-v0"
DIRECTORY = "Matsuki_2022_data"

OBSERVATION_SPACE_AFTER_PREPROCESSING = (
    2 if ENV_NAME == "CartPole-v1" else
    4 if ENV_NAME == "Acrobot-v1" else
    1 if ENV_NAME == "MountainCar-v0" else
    None
)

LR = (
    0.001 if ENV_NAME == "CartPole-v1" else
    0.005 if ENV_NAME == "Acrobot-v1" else
    0.001 if ENV_NAME == "MountainCar-v0" else
    None
)

# Configuration parameters for the whole setup
seed = 42
gamma = 0.99  # The discount rate γ is 0.99
epsilon = 0.5  # ε is set to 0.5 at the beginning of training 
epsilon_min = 0.01 # become 0.01 until the 401th episode
epsilon_max =  0.5 # Maximum epsilon greedy parameter
#epsilon_interval = epsilon_max - epsilon_min  # Rate of decay for random actions
batch_size = 256  # ND = 256 experiences are randomly sampled
max_steps_per_episode = 200 # An episode ends [...] when 200 steps have elapsed
max_episodes = 500  # Limit training episodes, will run until solved if smaller than 1

# Use the Atari environment
env = gym.make(ENV_NAME)

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

#env.seed(seed)
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


# Create models
model = QNetwork()
model_target = QNetwork()
model_target.load_state_dict(model.state_dict())
model_target.eval()

# AMSGrad with a momentum of 0.9. The learning rate is 0.001 for CartPole
optimizer = optim.Adam(model.parameters(), lr=LR, amsgrad=True)


# Reservoir
# The number of the reservoir neurons is Nx = 50 for each task
# connect recurrently with a connection probability p = 0.1
# g is set to 0.9
# W_in is generated from a uniform distribution between −1 and 1
# bias vector b is generated from a uniform distribution between -0.2 and 0.2
reservoir = Reservoir(
    units=50,
    rc_connectivity=0.1,
    sr=0.9,
    Win=np.random.uniform(-1, 1, (50, OBSERVATION_SPACE_AFTER_PREPROCESSING)),
    bias=np.random.uniform(-0.2, 0.2, (50,1)),
    activation="tanh",
    seed=seed
    )
# save the reservoir since it is randomly initialized and is not changed during the optimization
with open(f"{DIRECTORY}/{ENV_NAME}/Matsuki_2022_Reservoir.pkl", "wb") as rs:
    pickle.dump(reservoir, rs)


# Replay buffer
action_history = []
state_history = []
state_next_history = []
rewards_history = []
done_history = []
episode_reward_history = []
running_reward = 0
episode_count = 0
#epsilon_random_frames = 50000
#epsilon_greedy_frames = 1000000.0
max_memory_length = 10_000 # maximum size N = 10000
#update_after_actions = 4
# Once every two episodes, the parameters θi of the main network is copied to the parameters θ− of the target network
update_target_network = 2

save_model_episodes = 100

# Loss function
loss_function = nn.SmoothL1Loss()  # Huber Loss

consecutive_wins = 0

while True:
    state, _ = env.reset()
    state = preprocess(state)
    state_reservoir = reservoir(state, reset=True)
    # concatenate the reservoir state and the observation
    state = torch.tensor(np.concatenate([state_reservoir, state], axis=1), dtype=torch.float32)
    episode_reward = 0
    frame_count = 1

    # ε is set to 0.5 at the beginning of training 
    # and decays every trial by multiplying the 400th root of 0.02
    # to become 0.01 until the 401th episode
    epsilon *= np.power(0.02, 1/400)
    epsilon = max(epsilon, epsilon_min)

    for timestep in range(1, max_steps_per_episode):
        frame_count += 1

        # Epsilon-greedy action selection
        if epsilon > np.random.rand(1)[0]:
            action = np.random.choice(num_actions)
        else:
            with torch.no_grad():
                action = model(state).argmax(1).item()

        # Step environment
        state_next, reward, done, _, _ = env.step(action)
        state_next = preprocess(state_next)
        state_next_reservoir = reservoir(state_next, reset=False)
        state_next = torch.tensor(np.concatenate([state_next_reservoir, state_next], axis=1), dtype=torch.float32)
    
        # Update reward based on the conditions
        if ENV_NAME == "CartPole-v1":
            if done:
                reward = -1
        
        episode_reward += reward

        # Store transition in replay buffer
        action_history.append(action)
        state_history.append(state)
        state_next_history.append(state_next)
        rewards_history.append(reward)
        done_history.append(done)

        state = state_next

        # Update the model every `update_after_actions`
        if len(done_history) > batch_size:
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

    # if 200 steps are reached, the episode is considered successful
    if ENV_NAME == "CartPole-v1":
        if frame_count == max_steps_per_episode:
            consecutive_wins += 1
        else:
            consecutive_wins = 0
    
    elif ENV_NAME == "Acrobot-v1":
        if frame_count < max_steps_per_episode:
            consecutive_wins += 1
        else:
            consecutive_wins = 0
    
    elif ENV_NAME == "MountainCar-v0":
        if frame_count < max_steps_per_episode:
            consecutive_wins += 1
        else:
            consecutive_wins = 0

    episode_count += 1


    # If the agent has won 10 consecutive times, the task is considered solved
    if consecutive_wins >= 10:
        print(f"Solved at episode {episode_count}!")
        torch.save(model.state_dict(), f"{DIRECTORY}/{ENV_NAME}/Matsuki_2022_DQN_Final.pth")
        break
    
    # Update the target network every `update_target_network` episodes
    if episode_count % update_target_network == 0:
        model_target.load_state_dict(model.state_dict())
        print(f"Episode {episode_count}, epsilon {epsilon:.2f}, steps {frame_count}")
        # write training statistics to a csv file using pandas
        df = pd.DataFrame({
            "Episode": episode_count,
            "Epsilon": round(epsilon, 2),
            "Steps": frame_count
        }, index=[0])

        if not os.path.exists(f"{DIRECTORY}/{ENV_NAME}/training_stats.csv"):
            df.to_csv(f"{DIRECTORY}/{ENV_NAME}/training_stats.csv", index=False)
        else:
            df.to_csv(f"{DIRECTORY}/{ENV_NAME}/training_stats.csv", mode="a", header=False, index=False)
        



        


    if max_episodes > 0 and episode_count >= max_episodes:
        print(f"Stopped at episode {episode_count}!")
        break
