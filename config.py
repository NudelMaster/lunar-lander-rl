# config.py

# Environment parameters
ENV_NAME = 'LunarLander-v2'
ENV_STATE_DIM = 8  # LunarLander state space dimension
ENV_ACTION_DIM = 4  # LunarLander discrete actions count

# Training parameters
SCORE_TO_SOLVE = 200  # Score threshold to consider environment solved
MAX_EPISODES = 1500

# Hyperparameters
LEARNING_RATE = 1e-2
DISCOUNT_FACTOR = 0.99
NUM_ITER = 1500  # Number of iterations for training

