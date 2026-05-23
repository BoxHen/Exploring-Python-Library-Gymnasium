import gymnasium as gym
from   stable_baselines3 import PPO
from   stable_baselines3.common.evaluation import evaluate_policy
import os

print("==================================================")
print("  Gymnasium PPO Lab: CartPole-v1")
print("==================================================\n")

# Headless training
env = gym.make("CartPole-v1", render_mode=None)
# Init PPO Agent
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, n_steps=2048)
print("Training the PPO Neural Network for 10,000 timesteps...")
model.learn(total_timesteps=30000)
print("Training complete")
# Save training
model.save("ppo_cartpole_model")
env.close()

# Run visuals
print("\nOpening PyGame window to watch the trained PPO agent...")
eval_env = gym.make("CartPole-v1", render_mode="human")
# Check stability of agent by calculating avg score
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=3, render=True)
print(f"\nEvaluation Results over 3 episodes:")
print(f"Mean Reward: {mean_reward} (Max possible is 500.0)")
print(f"Standard Deviation: {std_reward}")
eval_env.close()