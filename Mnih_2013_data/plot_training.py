import pandas as pd
import matplotlib.pyplot as plt

# Load the training log
log = pd.read_csv('training_data.csv')

# Plot the training log
plt.figure(figsize=(10, 5))
plt.plot(log['frames'], log['avrg_reward'])
plt.xlabel('Frames')
plt.ylabel('Reward')
plt.title('Training Progress')


# Save the plot
plt.savefig('training_plot.png')


plt.show()