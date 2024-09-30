# Reinforcment Reservoirs

### Structure
The repository contains multiple training scripts and their resulting data. The paper by Chang and Futagami (2019) was successfully replicated and can be seen in the accordingly named files. <br>
The paper by Matsuki (2022) has been changed such that instead of the proposed tasks the game of Atari Breakout was used for training. This approach was not (yet) succesful, the (best) training can be seen in `RC_DQN.py`, but using the original approach based on the paper by Mnih et al. (2013) I was able to train the DQN with a very similar setup, such that it should be possible in theory to train the model using a Reservoir Computer.

### Report
A report / documentation of this project is available as the pdf file `reinforcement_reservoirs`. 

### Future Outlook
I hope that either me or someone else will be able to train the RC-DQN to play the Atari Breakout game eventually.