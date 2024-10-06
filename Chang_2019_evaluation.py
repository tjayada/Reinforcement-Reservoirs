import gymnasium as gym
from gymnasium.wrappers import ResizeObservation
import torch
import numpy as np
from reservoirpy.nodes import Reservoir
import pickle

# Define which model to load (e.g. "400" for the model trained for 400 episodes)
model_to_load = "0"
ENV_NAME = "CarRacing-v2"
# Define the path to the directory containing the models
DIRECTORY = "Chang_2019_data"


def normalize_input(image):
    """Normalize the input image to be between 0 and 1"""
    return image / 255.0


class CNN(torch.nn.Module):
    """Simple (untrained) CNN for projecting images to a lower-dimensional space"""
    def __init__(self):
        super(CNN, self).__init__()
        # Define the layers of the CNN
        self.cnn1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=31, stride=2, padding=15)
        self.cnn2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=14, stride=2, padding=6)
        self.cnn3 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=6, stride=2, padding=2)

        # Initialize the weights of the CNN 
        with torch.no_grad():
            torch.nn.init.normal_(self.cnn1.weight, mean=0.0, std=0.06)
            torch.nn.init.normal_(self.cnn2.weight, mean=0.0, std=0.06)
            torch.nn.init.normal_(self.cnn3.weight, mean=0.0, std=0.06)

        # Flatten the output of the CNN and pass it through a dense layer
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(in_features=8192, out_features=512)

        # Initialize the weights of the dense layer
        with torch.no_grad():
            torch.nn.init.normal_(self.linear1.weight, mean=0.0, std=0.1)

    def forward(self, x):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.flatten(x)
        x = self.linear1(x)       
        return x


class ReadoutNeuron(torch.nn.Module):
    """Simple readout neuron for mapping the reservoir state and cnn output to actions"""
    def __init__(self, input_size, reservoir_size, output_size):
        super(ReadoutNeuron, self).__init__()
        self.input_size = input_size  # D_u
        self.reservoir_size = reservoir_size  # D_x
        self.output_size = output_size  # D_y
        
        # Initialize W_out randomly, but we'll optimize it with CMA-ES later
        self.W_out = torch.nn.Parameter(torch.zeros(output_size, input_size + reservoir_size + 1), requires_grad=False)
        
    def forward(self, reservoir_state, input_vector):
        # Concatenate input vector, reservoir state, and bias term
        batch_size = reservoir_state.shape[0]
        bias = torch.ones((batch_size, 1), device=reservoir_state.device)  # Add bias term
        
        concatenated_input = torch.cat((reservoir_state, input_vector, bias), dim=1)  # Shape: (batch_size, D_x + D_u + 1)

        # Linear transformation using W_out
        output = torch.matmul(concatenated_input, self.W_out.T)  # Shape: (batch_size, D_y)

        return output


class TakeAction(object):
    """Class to preprocess the input, pass it through the CNN and Reservoir, and compute the action using the ReadoutNeuron"""
    def __init__(self, Preprocessing, CNN, Reservoir, Readout):
        super(TakeAction, self).__init__()
        self.pre = Preprocessing
        self.cnn = CNN
        self.reservoir = Reservoir
        self.readout = Readout

    def __call__(self, obs_input, cma_weights, is_first_frame):

        # Normalize the input image
        norm_in = self.pre(obs_input)

        # Pass the normalized input through the CNN
        model_out = self.cnn(norm_in)

        # Pass the output of the CNN through the reservoir
        reservoir_out = self.reservoir(model_out.flatten().detach().numpy(), reset=is_first_frame)
        reservoir_out = torch.tensor(reservoir_out, dtype=torch.float32)

        # Set the weights of the readout neuron
        with torch.no_grad():
            self.readout.W_out.copy_(torch.tensor(cma_weights.reshape(3, -1), dtype=torch.float32))

        # Compute the action using the readout neuron and the reservoir state and CNN output
        out = self.readout(model_out,reservoir_out).detach().numpy().flatten()
        out = np.tanh(out)

        # Clip the action to be between -1 and 1 according to the paper
        act = [ out[0], (out[1] + 1) / 2 , np.clip(out[2], 0, 1) ]
        act = np.float32(act)
        
        return act

# Load the trained CNN
cnn = CNN()
cnn.load_state_dict(torch.load(f"{DIRECTORY}/{ENV_NAME}/Chang_2019_CNN.pth", weights_only=True))
cnn.eval()

# Load the Reservoir
with open(f"{DIRECTORY}/{ENV_NAME}/Chang_2019_Reservoir.pkl", "rb") as f:
    reservoir = pickle.load(f)

# Load the best weights found by CMA-ES
with open(f"{DIRECTORY}/{ENV_NAME}/Chang_2019_Weigths_{model_to_load}.pkl", "rb") as f:
    best_w = pickle.load(f)

# Define the ReadoutNeuron
input_size = 512  # D_u
reservoir_size = 512  # D_x
output_size = 3  # D_y

# Initialize the ReadoutNeuron
readout = ReadoutNeuron(input_size, reservoir_size, output_size)

# Define the normalization function
preprocessing_function = normalize_input

# Initialize the TakeAction class
take_action = TakeAction(
                    Preprocessing=preprocessing_function, 
                    CNN=cnn, 
                    Reservoir=reservoir,
                    Readout=readout
                    )

# Play the game in human mode
env = gym.make(ENV_NAME, render_mode="human")
# Resize the observation to 64x64
env = ResizeObservation(env, 64)
# Reset the environment
obs, _ = env.reset()
# Reshape the observation to fit the CNN input
obs = torch.tensor(obs.reshape(3,64,64), dtype=torch.float32).unsqueeze(0)
# Define variables to keep track of the episode reward and whether the episode is done
done = False
episode_reward = 0
max_frames = 1_000
frame = 0

# Play the game
while (not done) and (frame < max_frames):
    # Get the action from the TakeAction class
    action = take_action(obs, best_w, frame == 0)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    # Accumulate the reward
    episode_reward += reward
    # Reshape the observation to fit the CNN input
    obs = torch.tensor(obs.reshape(3,64,64), dtype=torch.float32).unsqueeze(0)
    # Increment the frame counter
    frame += 1

# Print the episode reward
print(f"Evaluation Episode Reward: {episode_reward}")
env.reset()
env.close()