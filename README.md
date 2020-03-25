# left-shift

![A DQN agent reaching the 2048 tile.](docs/2048.gif)

This repository contains the code used in our project for the INF581: Advanced Topics in A.I. at École Polytechnique. We tackle the problem of training an agent to play the [game 2048](https://en.wikipedia.org/wiki/2048_(video_game)) with Reinforcement Learning. Our algorithm of choice is Deep Q-Learning (DQN).

## Project Structure

Below we detail the function of each directory:

* `agents`: contains scripts to train and evaluate agents (more details on the Running subsection), as well as the necessary code implementing custom callbacks and policies;
* `hyperparams`: contains YAML files detailing hyperparameters for agents (more details on the Hyperparameters subsection);
* `models`: contains pretrained agents;
* `utils`: contains auxiliary scripts for plotting results.

## Instalation

We recommend using a separate Python 3.7 environment for this project (there is an incompatibility issue when trying to load models created using Python 3.7 on other versions). Our dependencies are:

* [PyYAML](https://pyyaml.org);
* [OpenAI Gym](https://github.com/openai/gym);
* [TensorFlow 1.15.x](https://www.tensorflow.org) (2.x does not work);
* [Stable Baselines](https://github.com/hill-a/stable-baselines);
* [gym-text2048](https://github.com/mmcenta/gym-text2048).

A quick way to install them is to run the following command:
`pip install -r [requirements.txt|requirements-gpu.txt]`,
choosing the appropriate file for CPUs and GPUs.

To install the envronment, execute the following commands:
```
git clone https://github.com/mmcenta/gym-text2048
pip install -e gym-text2048
```

### Playing

Interactive player for humans:
```
python agents/play.py
```

Random agent:
```
python agents/random_agent.py
```

DQN using a 5-layer CNN:
```
python agents/dqn-cnn.py
```



