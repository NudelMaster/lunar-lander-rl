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
## Additional Notes for Google Colab Users

**⚠️ Warning:**  
This repository is designed to be run primarily from **Google Colab**.  
Running the setup and installation commands on your local machine may cause conflicts or mess up your system, especially due to system-level package installations required for Box2D support.  
If you wish to run locally, please carefully follow official installation instructions for Gym and its dependencies.
<a href="../../">
              <img class="farama-header__logo only-light" src="../../_static/img/gymnasium_black.svg" alt="Light Logo">
              <img class="farama-header__logo only-dark" src="../../_static/img/gymnasium_white.svg" alt="Dark Logo">
            <span class="farama-header__title">Gymnasium Documentation</span></a>

Google Colab runs on virtual machines that reset every session, so you need to install some system and Python packages each time you start a new session to run this project successfully, especially to support Gym’s Box2D environments and rendering.



Before running training or evaluation in Colab, execute this cell at the top of your notebook:

    # Run only in Google Colab or similar headless Linux environments
    !apt-get -qq install xvfb x11-utils &> /dev/null
    !pip install ufal.pybox2d --quiet
    !pip install pyvirtualdisplay moviepy pyglet PyOpenGL-accelerate --quiet
    !pip install numpy==1.23.5 matplotlib==3.7.0

After running the above, you may need to **restart your Colab runtime** to apply changes.

---


## Usage

### 1. Clone the Repository

    git clone https://github.com/your_username/lunar-lander-rl.git
    cd lunar-lander-rl

### 2. Install Dependencies

#### Local Machine

Install required Python packages with:

    pip install -r requirements.txt

For Gym’s Box2D support, you might need additional system packages depending on your OS. See the official Gym documentation for details.

<a href="https://gymnasium.farama.org/introduction/gym_compatibility/" target="_parent"><img src="https://gymnasium.farama.org/_static/img/gymnasium_black.svg" alt="Documentation"/></a>


#### Google Colab

Run the setup cell from the **Additional Notes for Google Colab Users** section above before running scripts.

---

### 3. Train the Agent

Run the training script to train the actor-critic agent on the LunarLander-v2 environment:

    python train_lunar_lander.py

Training will run for up to 1500 episodes and stop early if the average reward over the last 100 episodes reaches 200.

---

### 4. Evaluate and Render Final Policy

After training, run the evaluation script to watch the trained agent perform an episode and save the results as an animated GIF:

    python evaluate_lunar_lander.py

This will generate a file like `lunar_lander_final.gif` showing your agent’s landing.

