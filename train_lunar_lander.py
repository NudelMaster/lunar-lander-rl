
import torch, tqdm, numpy as np
from collections import deque



def run_training(agent, num_iter=1500, score_to_solve=200, pbar=None):
    # if using a gpu runtime DualNet().cuda() moves the policy to GPU
    # The optimizer will do the gradient updates for you
    # It needs the trainable parameters and a learning rate
    # feel free to experiment with other torch optimizers
    # Deque to store the last 100 episode returns
    device = agent.device
    net = agent.net
    env = agent.env
    optimizer = agent.optimizer
    discount_factor = agent.gamma
    action_dim = agent.action_dim

    episode_rewards = deque(maxlen = 100)
    # This progress_bar is useful to know how far along the training is
    start = agent.episodes_trained
    if pbar is None:
        progress_bar = tqdm.tqdm(
            range(start + num_iter),
            desc="Training",
            unit="ep")
    else:
        progress_bar = pbar

    for episode in progress_bar:
        optimizer.zero_grad()
        state = env.reset()
        done = False

        rewards = []
        state_values = []
        log_probs = []

        # Roll out one episode
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
            value = net(state_tensor, mode="critic")
            state_values.append(value)

            action_probs = net(state_tensor, mode="actor")
            action = np.random.choice(action_dim, p=action_probs.detach().cpu().numpy())
            log_prob = torch.log(action_probs[action])
            log_probs.append(log_prob)

            state, reward, done, _ = env.step(action)
            rewards.append(reward)

        # Episode finished
        ep_return = sum(rewards)
        episode_rewards.append(ep_return)
        agent.episodes_trained += 1

        
        # Early stopping if we have at least 100 returns
        if len(episode_rewards) == 100:
            avg_return = np.mean(episode_rewards)
            progress_bar.set_postfix_str(f"avg100={avg_return:.1f}")
            if avg_return >= score_to_solve:
                progress_bar.update(1)
                progress_bar.write(f"Environment solved in {agent.episodes_trained} episodes!")
                break

        # Compute discounted returns G_t
        T = len(rewards)
        returns = [0.0] * (T + 1)
        for t in reversed(range(T)):
            returns[t] = rewards[t] + discount_factor * returns[t + 1]
        returns = np.array(returns[:T])
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)

        # Compute actor & critic losses
        L_actor = torch.tensor(0.0, device=device)
        L_critic = torch.tensor(0.0, device=device)
        for t in range(T):
            delta_t = returns[t] - state_values[t]
            L_actor += -log_probs[t] * delta_t.detach()
            L_critic += delta_t ** 2
        L_critic *= 0.5
        loss = L_actor + L_critic

        loss.backward()

        optimizer.step()


    progress_bar.close()
