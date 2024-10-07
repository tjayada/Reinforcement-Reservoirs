import pandas as pd
import matplotlib.pyplot as plt


ENV_NAME = "CarRacing-v2"

# Load the training log
log = pd.read_csv(f'Chang_2019_data/{ENV_NAME}/training_stats.csv')

# Plot the training log
plt.figure(figsize=(10, 5))
plt.plot(log['Generation'], log['Reward'])
plt.xlabel('Generation')
plt.ylabel('Reward')
plt.title(f'{ENV_NAME}: Training Progress')


# Save the plot
plt.savefig(f'Chang_2019_data/{ENV_NAME}/training_plot.png')


plt.show()