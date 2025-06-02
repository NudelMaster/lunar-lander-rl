# LunarLander Reinforcement Learning Agent

This project implements an actor-critic reinforcement learning agent using PyTorch to solve the OpenAI Gym [LunarLander-v2](https://gym.openai.com/envs/LunarLander-v2/) environment. The agent learns to control the lunar lander to achieve stable landings with high rewards through policy gradient methods and value function approximation.

---

## Features

- Actor-Critic neural network architecture with separate policy (actor) and value (critic) heads  
- Training loop with reward tracking, return standardization, and early stopping on average reward  
- Supports GPU acceleration for faster training  
- Final policy evaluation episode rendering with frame capture and GIF export  
- Modular, easy-to-extend codebase for experimenting with other environments or RL algorithms

---

## Requirements

- Python 3.7+  
- PyTorch  
- OpenAI Gym  
- NumPy  
- Matplotlib  
- tqdm  
- imageio  

You can install dependencies using pip:

```bash
pip install torch gym numpy matplotlib tqdm imageio


