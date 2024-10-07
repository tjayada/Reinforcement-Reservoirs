import pandas as pd
import matplotlib.pyplot as plt

ENVS = ["CartPole-v1", "Acrobot-v1", "MountainCar-v0"]

for ENV_NAME in ENVS:
    # Load the training log
    log = pd.read_csv(f'Matsuki_2022_data/{ENV_NAME}/training_stats.csv')

    # Plot the training log
    plt.figure(figsize=(10, 5))
    plt.plot(log['Episode'], log['Steps'])
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title(f'{ENV_NAME}: Training Progress')

    # Save the plot
    plt.savefig(f'Matsuki_2022_data/{ENV_NAME}/training_plot.png')

    plt.show()