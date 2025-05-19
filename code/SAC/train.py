import gym
from stable_baselines3 import SAC  # Changed from PPO to SAC
from stable_baselines3.common.callbacks import BaseCallback
import csv
from env import CarlaEnv
import torch
from tqdm import tqdm  

class EpisodeCallback(BaseCallback):
    def __init__(self, total_episodes=10000, csv_file="training_rewards.csv", verbose=0):
        super(EpisodeCallback, self).__init__(verbose)
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.total_episodes = total_episodes
        self.csv_file = csv_file
        self.pbar = tqdm(total=self.total_episodes, desc="Training Progress")
        self.episode_count = 0

        with open(self.csv_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "reward", "episode_length", "termination_reason"])

    def _on_step(self):
        reward = self.locals['rewards'][0]
        done = self.locals['dones'][0]

        self.current_episode_reward += reward
        self.current_episode_length += 1

        if done:
            self.episode_count += 1
            termination_reason = self.locals['infos'][0].get('termination_reason', 'unknown')
            with open(self.csv_file, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([self.episode_count, self.current_episode_reward, self.current_episode_length, termination_reason])

            self.pbar.update(1)
            self.pbar.set_postfix({
                "Reward": f"{self.current_episode_reward:.2f}",
                "Length": self.current_episode_length
            })

            if self.episode_count % 100 == 0:
                self.model.save(f"models/sac_carla_{self.episode_count}.pth")

            self.current_episode_reward = 0
            self.current_episode_length = 0
            
            if self.episode_count >= self.total_episodes:
                self.pbar.close()
                return False
        
        return True

    def _on_training_end(self):
        self.pbar.close()

env = CarlaEnv()

model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    device="cuda:1" if torch.cuda.is_available() else "cpu",
    learning_rate=5e-5,
    buffer_size=1_000_000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    gradient_steps=1,
    policy_kwargs=dict(
        net_arch=dict(pi=[256, 256], qf=[256, 256])
    ),
    ent_coef='auto',
    train_freq=1,
    learning_starts=20000
)

callback = EpisodeCallback(total_episodes=10000)
model.learn(
    total_timesteps=int(1e8),  # SAC typically needs fewer timesteps
    callback=callback,
    log_interval=4
)
model.save("models/sac_carla_final.pth")
env.close()