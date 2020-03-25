# left-shift

![A DQN agent reaching the 2048 tile.](docs/2048.gif)

This repository contains the code used in our project for the INF581: Advanced Topics in A.I. at École Polytechnique. We tackle the problem of training an agent to play the [game 2048](https://en.wikipedia.org/wiki/2048_(video_game)) with Reinforcement Learning. Our algorithm of choice is Deep Q-Learning (DQN).

## Project Structure

Below we detail the function of each directory:

* `agents`: contains scripts to train and evaluate agents (more details on the Running subsection), as well as the necessary code implementing custom callbacks and policies;
* `docs`: contains the GIF you saw above and the final report of the project;
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
choosing the appropriate file depending on whether you wish to run the models on a CPU or a GPU.

To install the envronment, execute the following commands:
```
git clone https://github.com/mmcenta/gym-text2048
pip install -e gym-text2048
```
## Running

### Interactive player for humans:
```
python agents/play.py
```

### Random agent:
```
python agents/random_agent.py
```

### DQN:

#### To show an agent reaching 2048:
```
python agents/dqn.py --r2048
```

#### To demo the pretrained models:
```
python agents/dqn.py -mn MODEL_NAME --demo
```
##### Example:
```
python agents/dqn.py -mn cnn_5l_4_v2 --demo
```
The models are in the folder `models/`.
If the model has `_nohot`, you have to launch it with the `--no-one-hot` flag.

##### Example:
```
python agents/dqn.py -mn cnn_5l_4_v2_nohot --no-one-hot --demo
```

#### To launch a training:
```
python agents/dqn.py -mn MODEL_NAME --train
```

#### To launch a evaluation:
```
python agents/dqn.py -mn MODEL_NAME --eval
```

#### Complete usage:
```
python agents/dqn.py -h
```
```
usage: python agents/dqn.py [-h] [--env ENV] [--tensorboard-log TENSORBOARD_LOG]
              [--hyperparams-file HYPERPARAMS_FILE] [--model-name MODEL_NAME]
              [--n-timesteps N_TIMESTEPS] [--log-interval LOG_INTERVAL]
              [--hist-freq HIST_FREQ] [--eval-episodes EVAL_EPISODES]
              [--save-freq SAVE_FREQ] [--save-directory SAVE_DIRECTORY]
              [--log-directory LOG_DIRECTORY] [--seed SEED]
              [--verbose VERBOSE] [--no-one-hot] [--train] [--eval]
              [--extractor EXTRACTOR] [--demo] [--r2048]

optional arguments:
  -h, --help            show this help message and exit
  --env ENV             Environment id.
  --tensorboard-log TENSORBOARD_LOG, -tb TENSORBOARD_LOG
                        Tensorboard log directory.
  --hyperparams-file HYPERPARAMS_FILE, -hf HYPERPARAMS_FILE
                        Hyperparameter YAML file location.
  --model-name MODEL_NAME, -mn MODEL_NAME
                        Model name (if it already exists, training will be
                        resumed).
  --n-timesteps N_TIMESTEPS, -n N_TIMESTEPS
                        Number of timesteps.
  --log-interval LOG_INTERVAL
                        Log interval.
  --hist-freq HIST_FREQ
                        Dumps histogram each n steps.
  --eval-episodes EVAL_EPISODES
                        Number of episodes to use for evaluation.
  --save-freq SAVE_FREQ
                        Save the model every n steps (if negative, no
                        checkpoint).
  --save-directory SAVE_DIRECTORY, -sd SAVE_DIRECTORY
                        Save directory.
  --log-directory LOG_DIRECTORY, -ld LOG_DIRECTORY
                        Log directory.
  --seed SEED           Random generator seed.
  --verbose VERBOSE     Verbose mode (0: no output, 1: INFO).
  --no-one-hot          Disable one-hot encoding
  --train               Enable training
  --eval                Enable evaluation
  --extractor EXTRACTOR
                        Change extractor
  --demo                Enable rendering and runs for 1 episode
  --r2048               Show an agent reaching 2048
```

## Plotting training logs:

### To plot all the training logs:
```
python utils/plot_log_multi.py
```

### To plot a specific training log:
```
python utils/plot_log.py PATH_TO_LOG
```

#### Example:
```
python utils/plot_log.py logs/cnn_5l4_fc.npz
```

## Authors

- [Ahmed Bouhoula](https://github.com/bouhoula)
- [Matheus Castro](https://github.com/matheuscarius)
- [Matheus Centa](https://github.com/mmcenta)

