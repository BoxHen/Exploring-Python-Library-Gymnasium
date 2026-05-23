import gymnasium as gym
import numpy as np
import random
import time

env = gym.make("CartPole-v1", render_mode=None)

# Divide the continuous ranges into discrete buckets
# We care more about pole angle and velocity than the cart pos
num_buckets = (1, 1, 6, 6) # (Cart Pos, Cart Vel, Pole Angle, Pole Vel)

# Define upper and lower bounds
state_bounds = list(zip(env.observation_space.low, env.observation_space.high))
state_bounds[1] = [-0.5, 0.5]
state_bounds[3] = [-50, 50]

# Create a 4D Q-table matrix: [1][1][6][6][2 actions]
q_table = np.zeros(num_buckets + (env.action_space.n,))

# Helper function to convert continuous float states into bucket integers
def discretize_state(state):
    bucket_indices = []
    for i in range(len(state)):
        if state[i] <= state_bounds[i][0]:
            bucket_index = 0
        elif state[i] >= state_bounds[i][1]:
            bucket_index = num_buckets[i] - 1
        else:
            # Scale the float value into our bucket range
            bound_width = state_bounds[i][1] - state_bounds[i][0]
            offset = (state[i] - state_bounds[i][0]) / bound_width
            bucket_index = int(round((num_buckets[i] - 1) * offset))
        bucket_indices.append(bucket_index)
    return tuple(bucket_indices)

# Hyperparameters
learning_rate = 0.1
gamma = 0.99
epsilon = 1.0
decay_rate = 0.001
total_episodes = 10000

print("Training Tabular Q-Learning on CartPole using Buckets...")
for episode in range(total_episodes):
    raw_state, info = env.reset()
    state = discretize_state(raw_state)
    terminated, truncated = False, False

    while not (terminated or truncated):
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_raw_state, reward, terminated, truncated, info = env.step(action)
        next_state = discretize_state(next_raw_state)
        # Bellman Equation
        old_value = q_table[state + (action,)]
        next_max = np.max(q_table[next_state])
        q_table[state + (action,)] = old_value + learning_rate * (reward + gamma * next_max - old_value)
        state = next_state
        
    epsilon = max(0.01, epsilon - decay_rate)

env.close()
print("Training Complete!")

# Run visuals
gui_env = gym.make("CartPole-v1", render_mode="human")
for test_ep in range(2):
    raw_state, info = gui_env.reset()
    state = discretize_state(raw_state)
    terminated, truncated = False, False
    score = 0
    while not (terminated or truncated):
        action = np.argmax(q_table[state])
        raw_state, reward, terminated, truncated, info = gui_env.step(action)
        state = discretize_state(raw_state)
        score += reward
        time.sleep(0.02)
    print(f"Test Episode Score: {score}")
gui_env.close()