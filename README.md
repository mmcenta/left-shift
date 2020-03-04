## Dependencies
- [OpenAI Gym](https://github.com/openai/gym)
- [TensorFlow 1.x](https://www.tensorflow.org/install/pip) (2.x does not work)
- [Stable Baselines](https://github.com/hill-a/stable-baselines)

## Running

We suggest creating a virtual environment before installing the packages

### Installing dependencies

```
pip install gym
pip install tensorflow-gpu==1.15 (or pip install tensorflow==1.15 without GPU support)
pip install stable-baselines
```

### Installing gym environment

```
pip install -e gym_text2048
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



