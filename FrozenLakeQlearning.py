import gymnasium as gym
import numpy as np
import random
import time

print("================================================")
print("  Gymnasium Q-Learning Lab: FrozenLake-v1")
print("================================================\n")

# Fast training with epsilon decay fix
# Use render_mode=None to train at maximum CPU speed without UI lag
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode=None)

# Initialize Q Table to 16 states x 4 actions
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Hyperparameters
learning_rate = 0.1    # Alpha: learning rate
gamma = 0.95           # Gamma: discount factor for future rewards
epsilon = 1.0          # Start at 1.0 (100% exploration/random moves)
min_epsilon = 0.01     # Minimum exploration probability floor
decay_rate = 0.0015    # Amount to reduce epsilon after every single episode
total_episodes = 2000  # Number of training episodes

print(f"Phase 1: Training agent for {total_episodes} episodes in the background...")
print("Exploration rate (epsilon) will gradually decay from 1.0 down to 0.01.")

for episode in range(total_episodes):
    state, info = env.reset()
    terminated = False
    truncated = False

    while not (terminated or truncated):
        # Epsilon Greedy Strategy
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore: pick a random direction
        else:
            action = np.argmax(q_table[state])  # Exploit: pick best known move

        next_state, reward, terminated, truncated, info = env.step(action)
        # Update Q-Table via Bellman Equation
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        q_table[state, action] = old_value + learning_rate * (reward + gamma * next_max - old_value)
        state = next_state

    # Decay epsilon step-by-step after the episode concludes
    epsilon = max(min_epsilon, epsilon - decay_rate)

env.close()
print("Training complete. The Q Table has been mapped.")

# Run visuals
print("\nPhase 2: Opening PyGame window to observe the trained agent...")
print("The agent will now use 100% exploitation (epsilon=0) to show its perfect path.")

# Open the visual GUI environment
gui_env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="human")
action_names = ["Left", "Down", "Right", "Up"]

for test_episode in range(3):
    state, info = gui_env.reset()
    terminated = False
    truncated = False
    step_count = 0
    print(f"\n--- Running Test Episode {test_episode + 1} ---")
    
    while not (terminated or truncated):
        # Trained agent exploitation mode: zero randomness
        action = np.argmax(q_table[state]) 
        print(f"Step {step_count}: Agent is at State {state} -> Decided to move {action_names[action]}")
        state, reward, terminated, truncated, info = gui_env.step(action)
        step_count += 1
        # Pause for half a second so we can observe
        time.sleep(0.5) 
        
    if reward == 1.0:
        print("Success, the agent navigated the ice to the gift box.")
    else:
        print("Failed, The agent fell into a hole.")

gui_env.close()

print("\n================================================")
print("  Final Learned Q-Table Matrix")
print("================================================")
print("Rows = States (0 to 15), Columns = Actions (0=Left, 1=Down, 2=Right, 3=Up)\n")
print(np.round(q_table, 3))
print("\n================================================")