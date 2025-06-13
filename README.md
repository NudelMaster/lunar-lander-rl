# Lunar Lander Reinforcement Learning Agent

This project implements an actor-critic reinforcement learning agent using PyTorch to solve the OpenAI Gym [LunarLander-v2](https://gym.openai.com/envs/LunarLander-v2/) environment. The agent learns to control the lunar lander to achieve stable landings with high rewards through policy gradient methods and value function approximation.

---

## Features

- Actor-Critic neural network architecture with separate policy (actor) and value (critic) heads  
- Training loop with reward tracking, return standardization, and early stopping on average reward  
- Supports GPU acceleration for faster training  
- Final policy evaluation episode rendering with frame capture and GIF export  
- Modular, easy-to-extend codebase for experimenting with other environments or RL algorithms  
- Resumable progress bar using `tqdm` (continues from previous episode count)  

---

## Requirements

This project depends on the following Python packages and system libraries:

- `numpy`  
- `gym` (with Box2D support)  
- `matplotlib`  
- `pyvirtualdisplay`  
- `moviepy`  
- `torch` (PyTorch)  
- `tqdm`  

### Python package installation

Use the following command to install all necessary Python packages:

```bash
pip install -r requirements.txt
```


**⚠️ Warning:**  
This repository is designed to be run primarily from **Google Colab**.  
Running the setup and installation commands on your local machine may cause conflicts or mess up your system, especially due to system-level package installations required for Box2D support.  
If you wish to run locally, please carefully follow official installation instructions for Gym and its dependencies.

Google Colab runs on virtual machines that reset every session, so you need to install some system and Python packages each time you start a new session to run this project successfully, especially to support Gym’s Box2D environments and rendering.



<a href="https://colab.research.google.com/github/NudelMaster/lunar-lander-rl/blob/main/notebooks/colab_run.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

For Gym’s Box2D support, you might need additional system packages depending on your OS. See the official Gym documentation for details.

<a href="https://gymnasium.farama.org/introduction/gym_compatibility/" target="_parent">
<img src="https://gymnasium.farama.org/_static/img/gymnasium_white.svg" width = "50" alt="Documentation"/>
<img src="https://gymnasium.farama.org/_static/img/gymnasium_black.svg" width = "50" alt="Documentation"/>
</a>


### Lunar Lander Reinforcement Learning Videos

- [Before Training (Sample Video)](videos/untrained_video.gif)

---


- [After Training (Trained Model Video)](videos/trained_video.gif)

