# Reinforcement Reservoirs
<br>

## Introduction
This repository contains several training and evaluation scripts related to reinforcement learning and reservoir computing, as well as the resulting data. 
Neither the paper by Chang and Futagami (2019) nor the paper by Matsuki (2022) published code, so this repository may be the first to actually replicate these papers and provide the corresponding implementations. 
<br>
<br>

## Good to know
It is recommended to use the `requirements.txt` to build the python environment, but if someone decides against it, they must use `numpy version 1.24`, otherwise the `gymnasium` library will throw an error when running the `CarRacing-v2` environment.
And even if the `requirements.txt` is used, you will need to manually install and accept sublibraries such as <br>

	pip install gymnasium[box2d]
for the `CarRacing-v2` environment and

	pip install gymnasium[atari]
	pip install gymnasium[accept-rom-licence]
for the Atari `BreakoutNoFrameskip-v4` environment.

Finally, it should be mentioned that the `Reservoir` from the `reservoirpy` library does not have a save function, so I used `pickle` to serialise the reservoirs, but you have to set a `seed` for this to work, otherwise the reservoir will be randomly initialised when it is loaded again. This was quite counterintuitive to me, and I hope it will save someone's time when trying to figure out why training and evaluation performance is very different even though the same reservoir has been used.

<br>


## Repository Structure 
This repository is divided into 4 parts :

1. Chang and Futagami (2019)
2. Toshitaka Matsuki (2022)
3. Mnih et al. (2013) 
4. Combine all

Each part has corresponding training and evaluation scripts and corresponding directories containing the trained models and results.

### Chang and Futagami (2019)
The paper by Chang and Futagami (2019) was successfully replicated using the `Chang_2019_training.py` script and then evaluated using `Chang_2019_evaluation.py`. <br>
The resulting data can be found in the `Chang_2019_data` directory. 
<br>
<br>

An evaluation run is shown below :

![](images/Chang_2019_eval_run.gif)

<br>

Reference to the original paper:

```
@misc{chang2019reinforcementlearningconvolutionalreservoir,
      title={Reinforcement Learning with Convolutional Reservoir Computing}, 
      author={Hanten Chang and Katsuya Futagami},
      year={2019},
      eprint={1912.04161},
      archivePrefix={arXiv},
      primaryClass={cs.NE},
      url={https://arxiv.org/abs/1912.04161}, 
}
```

<br>

--

### Toshitaka Matsuki (2022)

The paper by Matsuki (2022) was successfully replicated using the `Matsuki_2022_training.py` script and then evaluated using `Matsuki_2022_evaluation.py`. <br>
The resulting data can be found in the `Matsuki_2022_data` directory.

<br>
Reference to the original paper:

```
@misc{matsuki2022deepqnetworkusingreservoir,
      title={Deep Q-network using reservoir computing with multi-layered readout}, 
      author={Toshitaka Matsuki},
      year={2022},
      eprint={2203.01465},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2203.01465}, 
}
```
<br>

--

### Mnih et al. (2013) 

Since the chosen environments by Matsuki (2022) may not provide adequate proof of the model working correctly, as these environment have been critized for their lack of complexity before, I have decided to implement a version of this model such that it can solve the Atari Breakout game. 
The decision of this environment is mainly based on the paper by Mnih et al. (2013) in which they first introduced DQNs sucesfully to the Atari games. So I first (partially) replicated the original findings using the `Mnih_2013_training.py` script then evaluated it using `Mnih_2013_evaluation.py`. <br>
The resulting data can be found in the `Mnih_2013_data` directory. <br> The training script itself is largely based on this [script](https://github.com/keras-team/keras-io/blob/master/examples/rl/deep_q_network_breakout.py) in a repository maintained by the keras team. 

<br>

--

### Combine all 

I then tried to modify the `Mnih_2013_training.py` script to incorporate the results of Chang and Futagami (2019) by using an untrained CNN to project the game frames into the reservoir, and the results of Matsuki (2022) by using Q-learning and an MLP to learn the task. This unsuccessful attempt can be seen in the training script `RC_DQN.py`. It seems that the "simple" method of projecting the game frames into the reservoir using an untrained CNN is not sophisticated enough, but many other hyperparameters could and probably should be adjusted to improve the performance of the model as well.

<br>

## Report
A report / documentation of this project is available as the `reinforcement_reservoirs` pdf file. 

<br>

## Quick demo and walktrough

There is a Google Collab notebook at :
<br>
[https://colab.research.google.com/drive/1SIlN2_4dq5Tw3hfE4W0am9xnUyEjcRkp?usp=sharing](https://colab.research.google.com/drive/1SIlN2_4dq5Tw3hfE4W0am9xnUyEjcRkp?usp=sharing) <br>
which gives a quick introduction to the topic, which can be useful if the number of scripts seems overwhelming.

<br>

## Future Outlook
I hope that someday the RC-DQN model will be able to play the Atari Breakout game. We will see.
