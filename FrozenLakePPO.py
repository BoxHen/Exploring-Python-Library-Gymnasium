import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import time

print("==================================================")
print("  Gymnasium PPO Lab: FrozenLake-v1")
print("==================================================\n")

# Headless training
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode=None)
# Init PPO Agent
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, n_steps=512)
print("Training PPO on a discrete 4x4 grid...")
model.learn(total_timesteps=40000) # 40,000 timesteps so neural network layers stabilize
print("Training complete")
env.close()

# Run visuals
print("\nOpening PyGame window to watch the PPO agent cross the ice...")
eval_env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="human")
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=3, render=True)
print(f"\nEvaluation Results over 3 episodes:")
print(f"Mean Reward: {mean_reward} (1.0 means 100% success reaching the gift)")
eval_env.close()