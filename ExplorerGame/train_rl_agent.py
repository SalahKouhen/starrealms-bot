import os
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from gym_wrapper import ExplorerOnlyGymEnv
from benchmark_ai import RandomAI, GreedyTradeAI, GreedyCombatAI, BalancedAI

# Create the output directory
models_dir = "../models"
logs_dir = "../logs"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

# Custom callback for training information
class TrainingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self):
        # Check if episode has ended
        if self.locals.get('dones')[0]:
            # Store the episode reward and length
            episode_reward = self.locals.get('rewards')[0]
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(self.n_calls - sum(self.episode_lengths))
            
            # Print episode info
            mean_reward = np.mean(self.episode_rewards[-100:])
            mean_length = np.mean(self.episode_lengths[-100:])
            print(f"Episode: {len(self.episode_rewards)}, " 
                  f"Reward: {episode_reward:.2f}, "
                  f"Mean Reward (100): {mean_reward:.2f}, "
                  f"Mean Length (100): {mean_length:.2f}")
        return True

def train_agent(total_timesteps=1000000, save_freq=10000):
    # Create the environment
    env = ExplorerOnlyGymEnv()
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    # Initialize the agent
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log=logs_dir
    )
    
    # Set up callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=models_dir,
        name_prefix="ppo_explorer"
    )
    training_callback = TrainingCallback()
    
    # Train the agent
    start_time = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, training_callback]
    )
    total_time = time.time() - start_time
    
    # Save the final model
    model.save(os.path.join(models_dir, "final_model"))
    env.save(os.path.join(models_dir, "vec_normalize.pkl"))
    
    print(f"Training completed in {total_time:.2f} seconds")
    return model

def evaluate_agent(model_path, episodes=10):
    # Load the trained model
    model = PPO.load(model_path)
    
    # Create a new environment for evaluation
    env = ExplorerOnlyGymEnv(render_mode="ansi")
    env = Monitor(env)
    
    # Run evaluation episodes
    total_rewards = 0
    wins = 0
    
    for i in range(episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            episode_reward += reward
            
            # Render the environment
            env.render()
        
        total_rewards += episode_reward
        if episode_reward > 0:
            wins += 1
        
        print(f"Episode {i+1}/{episodes}, Reward: {episode_reward:.2f}, Win: {episode_reward > 0}")
    
    print(f"Average reward: {total_rewards/episodes:.2f}")
    print(f"Win rate: {wins/episodes:.2f}")

def benchmark_against_ai(model_path, episodes_per_ai=100):
    """Compare RL agent performance against benchmark AIs."""
    # Load the trained model
    model = PPO.load(model_path)
    
    # Create AIs to benchmark against
    benchmark_ais = [
        RandomAI(),
        GreedyTradeAI(),
        GreedyCombatAI(),
        BalancedAI()
    ]
    
    # Evaluate against each AI
    for ai in benchmark_ais:
        print(f"\n=== Evaluating against {ai.name} AI ===")
        wins = 0
        
        # Create environment for this evaluation
        env = ExplorerOnlyGymEnv()
        env = Monitor(env)
        
        # Override the opponent's decision-making in the environment
        # This requires modifying your environment to accept an opponent AI
        env.game.opponent_ai = ai
        
        for i in range(episodes_per_ai):
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            
            step = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _, _ = env.step(action)
                episode_reward += reward
                step += 1
            
            if episode_reward > 0:
                wins += 1
                
            if (i+1) % 10 == 0:
                print(f"Progress: {i+1}/{episodes_per_ai} games")
        
        win_rate = wins / episodes_per_ai
        print(f"Win rate against {ai.name} AI: {win_rate:.2f}")
    
    print("\n=== Benchmark Summary ===")
    print(f"Model: {model_path}")
    print("Agent performed well against AIs with the following strategies:")
    # Additional analysis could go here

if __name__ == "__main__":
    mode = input("Select mode (train/evaluate/benchmark): ").strip().lower()
    
    if mode == "train":
        timesteps = int(input("Enter training timesteps (default: 1000000): ") or "1000000")
        train_agent(total_timesteps=timesteps)
    elif mode == "evaluate":
        model_path = input("Enter model path (e.g., ../models/final_model): ").strip()
        if not model_path:
            model_path = os.path.join(models_dir, "final_model")
        episodes = int(input("Enter number of evaluation episodes (default: 10): ") or "10")
        evaluate_agent(model_path, episodes=episodes)
    elif mode == "benchmark":
        model_path = input("Enter model path (e.g., ../models/final_model): ").strip()
        if not model_path:
            model_path = os.path.join(models_dir, "final_model")
        episodes = int(input("Enter episodes per benchmark AI (default: 100): ") or "100")
        benchmark_against_ai(model_path, episodes_per_ai=episodes)
    else:
        print("Invalid mode selected")