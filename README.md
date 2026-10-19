# Pong World Model

A lightweight Pong environment built from scratch in Python as a foundation for experimenting with **world models** — learning to predict how an environment evolves from visual observations and actions.

## Overview

This project implements a custom 32 × 32 Pong environment with paddle controls, ball dynamics, collision handling, and RGB observation rendering.

The goal is to use this environment to collect interaction data and develop a model that learns the environment's dynamics and predicts future states.

## Features

- Custom Pong environment implemented using NumPy
- 32 × 32 RGB observations
- Three discrete actions: left, stay, and right
- Ball movement and wall collision handling
- Paddle movement and ball-paddle collision detection
- Seeded random initialization for reproducibility
- Episode termination based on a maximum step limit

## Tech Stack

- Python
- NumPy
- Jupyter Notebook

## Project Structure

```text
pong-world-model/
├── pong_game_env.py        # Custom Pong environment
├── pong_worldmodel.ipynb   # World-model experiments
└── README.md
```

## Environment Specifications

| Property | Value |
|---|---|
| Observation dimensions | 32 × 32 × 3 |
| Number of actions | 3 |
| Action `0` | Move left |
| Action `1` | Stay |
| Action `2` | Move right |
| Maximum episode length | 120 steps |
| Observation data type | `float32` |

The environment provides an initial observation through `reset()` and returns the next observation and a `done` flag through `step(action)`.

## Installation

Clone the repository:

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>
```

Install the required dependency:

```bash
pip install numpy
```

For notebook experiments, install Jupyter if it is not already available:

```bash
pip install jupyter
```

## Usage

Import and initialize the environment:

```python
from pong_game_env import My_pong

env = My_pong(seed=42)

obs = env.reset()

print("Observation shape:", obs.shape)
```

Take an action:

```python
action = env.RIGHT

next_obs, done = env.step(action)

print("Next observation shape:", next_obs.shape)
print("Episode finished:", done)
```

Reset after an episode ends:

```python
if done:
    obs = env.reset()
```

## World Model Development

The project aims to learn a predictive model of the Pong environment from collected interaction data.

Planned workflow:

1. **Data collection:** Record observations, actions, and next observations.
2. **Representation learning:** Encode RGB observations into compact latent states.
3. **Dynamics learning:** Predict future latent states conditioned on actions.
4. **Prediction:** Use the learned dynamics to predict future environment states.
5. **Evaluation:** Compare predicted states or decoded frames with actual outcomes.

## Current Progress

- [x] Implement the basic Pong environment
- [x] Define paddle controls and ball movement
- [x] Render RGB observations
- [x] Implement wall and paddle collision handling
- [x] Add a maximum episode step limit
- [ ] Collect transition data
- [ ] Train the world model
- [ ] Evaluate future-state predictions

## Future Work

- Develop a transition dataset generation pipeline
- Train an image encoder and decoder
- Train a recurrent dynamics model
- Visualize predicted and actual future frames
- Evaluate prediction accuracy over multiple time steps


