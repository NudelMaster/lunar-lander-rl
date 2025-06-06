import torch
def evaluate(agent, greedy: bool = True, score_to_solve: float = 200.0):
    # Run one episode greedily, render frames, save gif


    
    env = agent.env
    net = agent.net
    device = agent.device      
    state = env.reset()
    done = False
    total_reward = 0
    net.eval()  # Set the network to evaluation mode

    frames = []
    
    while not done:
        frames.append(env.render(mode='rgb_array'))

        s_tensor = torch.tensor(state, dtype=torch.float32).to(device)
        probs = net(s_tensor, mode='actor')
        if greedy:
            # Choose the action with the highest probability
            action = torch.argmax(probs).item()
        else:
            # Sample an action from the probability distribution
            action = torch.multinomial(probs, num_samples=1).item()
        state, reward, done, _ = env.step(action)
        total_reward += reward



    return [] if total_reward < score_to_solve else frames, total_reward
    

