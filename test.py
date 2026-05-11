import time

import gymnasium as gym
# from highway_env.envs import TwoLaneOvertakingEnv
from overtaking_environment import register_two_lane_overtaking_env

register_two_lane_overtaking_env()

# Create the environment
env = gym.make('TwoLaneOvertaking-v0', render_mode='human')  # Use render_mode='human' to visualize
obs, info = env.reset()

print("Environment created successfully")
print(f"Initial observation shape: {obs.shape}")

# Test discrete actions
print("\nTesting discrete actions:")
discrete_actions = [0, 1, 2, 3, 4]  # SLOWER, IDLE, FASTER, LEFT, RIGHT
for action in discrete_actions:
    # run this action for 1000 steps (reset if episode ends)
    for step_idx in range(1000):
        obs, reward, terminated, truncated, info = env.step(action)
        if (step_idx + 1) % 100 == 0:
            print(f"Action {action} step {step_idx+1}: reward={reward:.2f}, terminated={terminated}, truncated={truncated}")
        if terminated or truncated:
            obs, info = env.reset()
        time.sleep(0.01)  # add a small delay to better visualize the environment

env.close()
print("\nAll tests completed successfully!")
