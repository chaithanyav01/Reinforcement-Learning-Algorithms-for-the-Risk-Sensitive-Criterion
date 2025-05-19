import gym
from stable_baselines3 import SAC  # Changed from PPO to SAC
from env import CarlaEnv
import csv
from tqdm import tqdm
import torch 



device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Load the trained SAC model
model = SAC.load("models/sac_carla_5000.pth",device=device)  # Update with your model path

# Initialize the environment
env = CarlaEnv()

# CSV file to save episode results
csv_file = "inference_results.csv"  # Changed file name for clarity
with open(csv_file, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(["episode", "cumulative_reward", "episode_length", "termination_reason"])
print("csv_file ready")

# Run inference for specified episodes
total_episodes = 500
for episode in tqdm(range(total_episodes), desc="Running Episodes", unit="episode"):
    obs = env.reset()
    done = False
    cumulative_reward = 0
    episode_length = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        cumulative_reward += reward
        episode_length += 1

    # Get termination reason from info
    termination_reason = info.get("termination_reason", "unknown")

    # Save episode results to CSV
    with open(csv_file, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([episode + 1, cumulative_reward, episode_length, termination_reason])

env.close()
print("SAC Inference completed. Results saved to", csv_file)