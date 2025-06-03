import torch,gym, numpy as np, matplotlib.pyplot as plt
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
from dual_net import DualNet
from config import (
    ENV_NAME,
    ENV_STATE_DIM,
    ENV_ACTION_DIM,
    SCORE_TO_SOLVE,
    LEARNING_RATE,
    DISCOUNT_FACTOR,
    NUM_ITER   
)
from train_lunar_lander import run_training
from eval import evaluate
from tqdm import tqdm
class Agent:
    def __init__(
        self,
        env_name=ENV_NAME,
        state_dim=ENV_STATE_DIM,
        action_dim=ENV_ACTION_DIM,
        lr=LEARNING_RATE,
        gamma=DISCOUNT_FACTOR,
        num_iter=NUM_ITER
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = gym.make(env_name)
        self.gamma = gamma
        self.action_dim = action_dim
        self.lr = lr

        # Initialize the actor-critic network
        self.net = DualNet(state_dim, action_dim).to(self.device)

        # Optimizer for the network
        self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr=self.lr)

        # Track how many episodes we've trained so far
        self.episodes_trained = 0

        
        

    def train(
        self,
        max_episodes: int = NUM_ITER,
        score_to_solve: int = SCORE_TO_SOLVE,
        resume: bool = True,
        save_path: str = "lunar_lander_actor_critic.pth",
        learning_rate: float = LEARNING_RATE,
        num_iter: int = NUM_ITER

    ):
        if not resume or self.episodes_trained == 0:
            print("Starting fresh training...")
            self.reset()

        # override the learning rate if requested
        # Override learning rate if requested
        if learning_rate is not None:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = learning_rate

        # Build a single tqdm bar that spans from 0 → (already_done + max_episodes)
        already_done = self.episodes_trained
        total_target = already_done + num_iter
        pbar = tqdm(total=total_target, desc="Training", unit="ep")
        pbar.update(already_done)
        
        run_training(
            agent=self,
            max_episodes=max_episodes,
            score_to_solve=score_to_solve,
            progress_bar = pbar
        )
        self.save_model(save_path)
        
                     
    def save_model(self, path):
        ''' Save the model state to a file '''
        assert path is not None, "Path to model cannot be None"
        torch.save(self.net.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        ''' Load the model state from a file '''
        assert path is not None, "Path to model cannot be None"
        self.net.load_state_dict(torch.load(path, map_location=self.device))
        self.net.to(self.device)
        print(f"Model loaded from {path}")


    def reset(self):
        self.episodes_trained = 0
        def _reset_weights(module):
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

        self.net.apply(_reset_weights)

        print("Environment reset and training counter cleared.")
            
         
    def evaluate(self, model_path: str, greedy: bool = True):
        self.load_model(model_path)
        frames, score = evaluate(agent = self, greedy = greedy)
        if frames:
            print(f"Evaluation score: {score}")
            return self.visualize_trajectory(frames)
        else:
            print("Score below threshold, no frames to visualize.")
            return None

    def visualize_trajectory(self, frames, fps=50):
        duration = int(len(frames) // fps + 1)
        fig, ax = plt.subplots()
        def make_frame(t, ind_max=len(frames)):
            ax.clear()
            ax.imshow(frames[min((int(fps*t),ind_max-1))])
            return mplfig_to_npimage(fig)
        plt.close()
        return VideoClip(make_frame, duration=duration)





    
