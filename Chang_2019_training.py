import numpy as np
from cmaes import CMA
import torch
import reservoirpy as rpy
from reservoirpy.nodes import Reservoir
import warnings
import pickle
import pandas as pd
import os

ENV_NAME = "CarRacing-v2"
DIRECTORY = "Chang_2019_data"

warnings.filterwarnings("ignore")
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device(
    "cuda" if torch.cuda.is_available() else 
    "mps" if torch.backends.mps.is_available() else
    "cpu")


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
        reservoir_out = self.reservoir(model_out.flatten().detach().cpu().numpy(), reset=is_first_frame)
        reservoir_out = torch.tensor(reservoir_out, dtype=torch.float32)

        # Set the weights of the readout neuron
        with torch.no_grad():
            self.readout.W_out.copy_(torch.tensor(cma_weights.reshape(3, -1), dtype=torch.float32))

        # Compute the action using the readout neuron and the reservoir state and CNN output
        out = self.readout(model_out,reservoir_out).detach().cpu().numpy().flatten()
        out = np.tanh(out)

        # Clip the action to be between -1 and 1 according to the paper
        act = [ out[0], (out[1] + 1) / 2 , np.clip(out[2], 0, 1) ]
        act = np.float32(act)
        
        return act


def worker(environment, take_action, weights):
        """Worker function to run the environment in parallel with the given weights and return the total reward"""
        total_reward = []
        # Run the environment 2 times and take the average reward (paper did 8 times)
        for _ in range(2):
            # Reset the environment for each run
            obs, _ = environment.reset()
            # Reshape the observation to match the input shape of the CNN
            obs = torch.tensor(obs.reshape(3,64,64), dtype=torch.float32).unsqueeze(0)
            # Set variables for the loop
            done = False
            episode_reward = 0
            max_frames = 1_000
            frame = 0
            # Run the environment until it is done or the maximum number of frames is reached
            while (not done) and (frame < max_frames):
                # Get the action from the take_action function, which preprocesses the input, passes it through the CNN and Reservoir, and computes the action
                action = take_action(obs, weights, frame == 0)
                obs, reward, terminated, truncated, info = environment.step(action)
                done = terminated or truncated
                # Accumulate the reward for the episode
                episode_reward += reward
                # Reshape the observation to match the input shape of the CNN
                obs = torch.tensor(obs.reshape(3,64,64), dtype=torch.float32).unsqueeze(0)
                # Increment the frame counter to break the loop if the maximum number of frames is reached
                frame += 1  
            # Append the total reward for the episode
            total_reward.append(episode_reward)
        # Return the weights and the negative mean of the total reward 
        # Negative mean is used because CMA-ES minimizes the objective function
        return (weights, - np.mean(total_reward))
        

if __name__ == "__main__":
    """Main function to run the CMA-ES optimization with the CarRacing-v2 environment"""
    from multiprocessing import Pool
    import gymnasium as gym
    from gymnasium.wrappers import ResizeObservation
    import tqdm
    
    save_model_episode = 100

    # Set the parameters for the CMA-ES optimization
    parallel_runs = 16
    generations = 500

    # Set the dimensions of the reservoir and the input and output sizes
    input_size = 512  # D_u
    reservoir_size = 512  # D_x
    output_size = 3  # D_y

    # Initialize the CMA-ES optimizer
    optimizer = CMA(
                    mean=np.random.normal(loc=0.0, scale=0.1, size= output_size * (input_size + reservoir_size + 1) ), 
                    sigma=1.0,
                    population_size=parallel_runs,
                    )

    # Initialize the CarRacing-v2 environment with a wrapper to resize the observation
    env = gym.make(ENV_NAME, render_mode="rgb_array")
    env = ResizeObservation(env, 64)
    
    # Initialize the CNN, Reservoir
    cnn = CNN().eval()
    reservoir = Reservoir(
        units=reservoir_size,
        sr=0.95,
        lr=0.8,
        seed=seed
        )
    # Save the CNN and Reservoir since they are randomly initialized and are not changed during the optimization
    torch.save(cnn.state_dict(), f'{DIRECTORY}/{ENV_NAME}/Chang_2019_CNN.pth')
    with open(f"{DIRECTORY}/{ENV_NAME}/Chang_2019_Reservoir.pkl", "wb") as rs:
        pickle.dump(reservoir, rs)
    
    # Initialize the ReadoutNeuron
    readout = ReadoutNeuron(
        input_size=input_size, 
        reservoir_size=reservoir_size, 
        output_size=output_size
        )
    
    # Set the preprocessing function to normalize the input image
    preprocessing_function = normalize_input
    
    # Initialize the TakeAction class with the preprocessing function, CNN, Reservoir, and ReadoutNeuron
    take_action = TakeAction(
                        Preprocessing=preprocessing_function, 
                        CNN=cnn, 
                        Reservoir=reservoir,
                        Readout=readout
                        )

    # Run the CMA-ES optimization for the specified number of generations
    for gen in tqdm.tqdm(range(generations)):
        # Use a multiprocessing pool to run the environment in parallel with the CMA-ES optimizer
        with Pool(parallel_runs) as pool:
            # List to store the solutions from the parallel workers
            solutions = []
            # Form the data to be passed to the workers
            data = [(env, take_action, optimizer.ask()) for i in range(parallel_runs)]
            # Run the workers in parallel
            solutions = pool.starmap(worker, data)
            # Update the optimizer with the solutions from the workers
            optimizer.tell(solutions)
            # Get the best solution from the optimizer
            best_solution = min(solutions, key=lambda s: s[1])
            # Print the average value of the solutions for the generation
            avg = np.mean([-s[1] for s in solutions])
            print(f"Generation {gen}, Average Value: {avg}")
            # Save the training progress using pandas
            df = pd.DataFrame({
                    "Generation": gen,
                    "Reward": avg,
                }, index=[0])

            if not os.path.exists(f"{DIRECTORY}/{ENV_NAME}/training_stats.csv"):
                df.to_csv(f"{DIRECTORY}/{ENV_NAME}/training_stats.csv", index=False)
            else:
                df.to_csv(f"{DIRECTORY}/{ENV_NAME}/training_stats.csv", mode="a", header=False, index=False)
            


            best_w = best_solution[0]

        if gen%save_model_episode == 0:
            # Save the best weights of the CMA-ES optimization for the ReadoutNeuron
            with open(f"{DIRECTORY}/{ENV_NAME}/Chang_2019_Weigths_{gen}.pkl", "wb") as bw:   
                pickle.dump(best_w, bw)

    # Save the final weights of the CMA-ES optimization for the ReadoutNeuron
    with open(f"{DIRECTORY}/{ENV_NAME}/Chang_2019_Weigths_Final.pkl", "wb") as bw:   
        pickle.dump(best_w, bw)
    
    # Close the environment
    env.close()
