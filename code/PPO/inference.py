import gym
from stable_baselines3 import PPO
from env import CarlaEnv
import csv
from tqdm import tqdm  # Import tqdm for progress bar
import torch 

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Load the trained model
model = PPO.load("models/ppo_carla_6800.pth",device=device)

# Initialize the environment
env = CarlaEnv()

# CSV file to save episode results
csv_file = "inference_results.csv"
with open(csv_file, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(["episode", "cumulative_reward", "episode_length", "termination_reason"])

# Run inference for 500 episodes
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
        # if episode_length % 100 == 0:
        #     print("Episode Length: ", episode_length)

    # Get termination reason from info
    termination_reason = info.get("termination_reason", "unknown")

    # Save episode results to CSV
    with open(csv_file, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([episode + 1, cumulative_reward, episode_length, termination_reason])

env.close()
print("Inference completed. Results saved to inference_results.csv.")