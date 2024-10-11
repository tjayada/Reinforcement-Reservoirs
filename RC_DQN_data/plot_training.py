import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ENV_NAME = "BreakoutNoFrameskip-v4"
DIRECTORY = "RC_DQN_data"

# Load the training log
log = pd.read_csv(f'{DIRECTORY}/{ENV_NAME}/training_stats.csv')


# Plot the training log
plt.figure(figsize=(10, 5))
plt.plot(log['frame'], log['average_reward'])
plt.xlabel('Frames')
plt.ylabel('Reward')
plt.title(f'{ENV_NAME}: Training Progress')


# Save the plot
plt.savefig(f'{DIRECTORY}/{ENV_NAME}/training_plot.png')


plt.show()