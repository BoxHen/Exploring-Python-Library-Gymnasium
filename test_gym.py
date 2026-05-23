import gymnasium as gym

env = gym.make("CartPole-v1")

observation, info = env.reset(seed=42)

for _ in range(10):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

print("Gymnasium installed successfully!")

env.close()