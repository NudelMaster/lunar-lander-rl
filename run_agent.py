from agent import Agent
from config import MAX_EPISODES, SCORE_TO_SOLVE

if __name__ == "__main__":
    agent = Agent()
    
    # Train from scratch:
    agent.train(max_episodes=MAX_EPISODES, score_to_solve=SCORE_TO_SOLVE, resume=False)
    
    # Evaluate (and save a GIF) afterward:
    total_reward = agent.evaluate(model_path="lunar_lander_actor_critic.pth", greedy=True)

    # Optionally, continue training later:
    # agent.train(max_episodes=500, resume=True)
