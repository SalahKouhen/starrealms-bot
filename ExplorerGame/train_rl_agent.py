import os
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium as gym
from gymnasium import spaces

from gym_wrapper import ExplorerOnlyGymEnv
from benchmark_ai import RandomAI, GreedyTradeAI, GreedyCombatAI, BalancedAI

# Set up directories
import os
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(base_dir, "models")
logs_dir = os.path.join(base_dir, "logs")

# Create the output directory
os.makedirs(models_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

# Custom callback for training information
class TrainingCallback(BaseCallback):
    def __init__(self, verbose=0, saved_episode_count=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = saved_episode_count  # Add this to track total episodes
        
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
            self.episode_count += 1  # Increment this counter
            
            # Use self.episode_count instead of len(self.episode_rewards)
            print(f"Episode: {self.episode_count}, " 
                  f"Reward: {episode_reward:.2f}, "
                  f"Mean Reward (100): {mean_reward:.2f}, "
                  f"Mean Length (100): {mean_length:.2f}")
        return True

def train_agent(total_timesteps=1000000, save_freq=10000, continue_from=None):
    # Add .zip extension if not present
    if continue_from and not continue_from.endswith('.zip'):
        continue_from = continue_from + '.zip'
    
    # Debug the model path (keeping only essential checks)
    print(f"Checking for model at: {continue_from}")
    print(f"File exists: {os.path.exists(continue_from) if continue_from else 'No path provided'}")
    
    # Create the environment
    env = ExplorerOnlyGymEnv()
    env = FlattenActionWrapper(env)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    
    # Normalize the environment
    env = VecNormalize(
        env, 
        norm_obs=True, 
        norm_reward=True,
        norm_obs_keys=[
            "p0_authority", "p0_trade_pool", "p0_combat_pool",
            "p1_authority", "p1_trade_pool", "p1_combat_pool"
        ]
    )
    
    # Load previous model if specified
    if continue_from and os.path.exists(continue_from):
        print(f"Continuing training from {continue_from}")
        try:
            # Try to load the model
            model = PPO.load(continue_from, env=env)
            print("✅ Model loaded successfully")
            
            # Also load normalization stats if available
            norm_path = os.path.join(os.path.dirname(continue_from), "vec_normalize.pkl")
            if os.path.exists(norm_path):
                print(f"Loading normalization parameters from {norm_path}")
                
                # Load the normalization parameters
                env = VecNormalize.load(norm_path, env)
                env.training = True  # Continue updating normalization statistics
                print("✅ Normalization parameters loaded successfully")
            else:
                print("❌ Normalization file not found!")
                
            # Do a test prediction with better error handling
            try:
                test_obs = env.reset()[0]
                print(f"Observation shape/type: {type(test_obs)}")
                if isinstance(test_obs, dict):
                    print(f"Observation keys: {test_obs.keys()}")
                elif isinstance(test_obs, np.ndarray):
                    print(f"Observation shape: {test_obs.shape}")
                
                print("About to call model.predict()...")
                action, _ = model.predict(test_obs, deterministic=True)
                print(f"Test prediction successful: {action}")
            except Exception as e:
                print(f"❌ Detailed error during prediction: {repr(e)}")
                import traceback
                traceback.print_exc()
                raise  # Re-raise to be caught by outer try/except
            
        except Exception as e:
            print(f"❌ Error loading model: {repr(e)}")
            print("Starting new model instead")
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
    else:
        print("Starting new model from scratch")
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
    
    # In train_agent function:
    if os.path.exists("../models/training_stats.npz"):
        print("Loading previous training statistics")
        stats = np.load("../models/training_stats.npz")
        saved_episode_count = int(stats['episode_count'])
        training_callback = TrainingCallback(saved_episode_count=saved_episode_count)
    else:
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
    
    # After training completes:
    np.savez("../models/training_stats.npz", 
             episode_count=len(training_callback.episode_rewards))
    
    print(f"Training completed in {total_time:.2f} seconds")
    return model

def evaluate_agent(model_path, episodes=10):
    # Load the trained model
    model = PPO.load(model_path)
    
    # Create a new environment for evaluation
    env = ExplorerOnlyGymEnv(render_mode="ansi")
    env = FlattenActionWrapper(env)  # Add this line
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
    # Load the trained model
    model = PPO.load(model_path)
    
    # Create only GreedyCombatAI to benchmark against
    benchmark_ais = [
        GreedyCombatAI()
    ]
    
    results = {}
    
    for ai in benchmark_ais:
        print(f"\n=== Evaluating against {ai.name} AI ===")
        
        # Test with agent as player 0
        print("\nTesting RL agent as player 0:")
        wins_as_p0 = 0
        
        for i in range(episodes_per_ai // 2):
            # Create fresh environment for each game
            base_env = ExplorerOnlyGymEnv()
            env = FlattenActionWrapper(base_env)
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            
            # RL agent is always player 0
            while not done:
                # If it's player 0's turn (RL agent)
                if base_env.game.current == 0:
                    action, _ = model.predict(obs, deterministic=True)
                else:
                    # Get action from benchmark AI
                    current_player = base_env.game.players[base_env.game.current]
                    opponent = base_env.game.players[1 - base_env.game.current]
                    ai_action = ai.get_action(current_player, opponent)
                    
                    # Convert AI action to flattened format expected by wrapper
                    action = np.zeros(7, dtype=np.int64)
                    
                    # Set play_cards (first 5 elements)
                    for idx in ai_action.get("play_cards", []):
                        if idx < 5:  # Only consider first 5 cards
                            action[idx] = 1
                    
                    # Set scrap_explorers and buy_explorers
                    action[5] = min(ai_action.get("scrap_explorers", 0), 5)  # Cap at 5
                    action[6] = min(ai_action.get("buy_explorers", 0), 5)    # Cap at 5
                
                # Take step in environment
                obs, reward, done, _, _ = env.step(action)
                episode_reward += reward
                
            winner = 0 if base_env.game.players[1].authority <= 0 else 1
            if winner == 0:  # RL agent won as player 0
                wins_as_p0 += 1
                
            if (i+1) % 10 == 0:
                print(f"Progress: {i+1}/{episodes_per_ai//2} games, Win rate: {wins_as_p0/(i+1):.2f}")
        
        # Test with agent as player 1
        print("\nTesting RL agent as player 1:")
        wins_as_p1 = 0
        
        for i in range(episodes_per_ai // 2):
            # Create fresh environment for each game
            base_env = ExplorerOnlyGymEnv()
            env = FlattenActionWrapper(base_env)
            obs, _ = env.reset()
            
            # Take one step with AI to make RL agent player 1
            current_player = base_env.game.players[0]
            opponent = base_env.game.players[1]
            ai_action = ai.get_action(current_player, opponent)
            
            # Convert AI action to flattened format
            action = np.zeros(7, dtype=np.int64)
            for idx in ai_action.get("play_cards", []):
                if idx < 5:
                    action[idx] = 1
            action[5] = min(ai_action.get("scrap_explorers", 0), 5)
            action[6] = min(ai_action.get("buy_explorers", 0), 5)
            
            # Take first step with AI
            obs, reward, done, _, _ = env.step(action)
            episode_reward = -reward  # Flip sign since we're player 1
            
            # Continue the game
            while not done:
                if base_env.game.current == 1:  # RL agent's turn
                    action, _ = model.predict(obs, deterministic=True)
                else:  # AI's turn
                    current_player = base_env.game.players[base_env.game.current]
                    opponent = base_env.game.players[1 - base_env.game.current]
                    ai_action = ai.get_action(current_player, opponent)
                    
                    # Convert to flattened format
                    action = np.zeros(7, dtype=np.int64)
                    for idx in ai_action.get("play_cards", []):
                        if idx < 5:
                            action[idx] = 1
                    action[5] = min(ai_action.get("scrap_explorers", 0), 5)
                    action[6] = min(ai_action.get("buy_explorers", 0), 5)
                
                obs, reward, done, _, _ = env.step(action)
                episode_reward -= reward  # Flip sign since we're player 1
            
            winner = 0 if base_env.game.players[1].authority <= 0 else 1
            if winner == 1:  # RL agent won as player 1
                wins_as_p1 += 1
                
            if (i+1) % 10 == 0:
                print(f"Progress: {i+1}/{episodes_per_ai//2} games, Win rate: {wins_as_p1/(i+1):.2f}")
        
        # Calculate overall results
        win_rate_p0 = wins_as_p0 / (episodes_per_ai // 2)
        win_rate_p1 = wins_as_p1 / (episodes_per_ai // 2)
        overall_win_rate = (wins_as_p0 + wins_as_p1) / episodes_per_ai
        
        results[ai.name] = {
            "as_p0": win_rate_p0,
            "as_p1": win_rate_p1,
            "overall": overall_win_rate
        }
        
        print(f"\nWin rate as player 0: {win_rate_p0:.2f}")
        print(f"Win rate as player 1: {win_rate_p1:.2f}")
        print(f"Overall win rate: {overall_win_rate:.2f}")
    
    return results

def debug_benchmark_against_ai(model_path, episodes_per_ai=3):
    """Run games with minimal output for easier debugging."""
    model = PPO.load(model_path)
    benchmark_ais = [GreedyCombatAI()]
    
    for ai in benchmark_ais:
        print(f"\n=== RL Agent vs {ai.name} AI ===")
        
        for rl_player_idx in [0, 1]:
            print(f"\n* RL Agent as Player {rl_player_idx} *")
            
            for i in range(episodes_per_ai):
                print(f"\n--- Game {i+1}/{episodes_per_ai} ---")
                
                base_env = ExplorerOnlyGymEnv(render_mode=None)  # No rendering
                env = FlattenActionWrapper(base_env)
                obs, _ = env.reset()
                done = False
                
                # If RL agent is player 1, have AI take first turn
                if rl_player_idx == 1:
                    current_player = base_env.game.players[0]
                    opponent = base_env.game.players[1]
                    ai_action = ai.get_action(current_player, opponent)
                    
                    action = np.zeros(7, dtype=np.int64)
                    for idx in ai_action.get("play_cards", []):
                        if idx < 5:
                            action[idx] = 1
                    action[5] = min(ai_action.get("scrap_explorers", 0), 5)
                    action[6] = min(ai_action.get("buy_explorers", 0), 5)
                    
                    # Summary line for AI's first turn
                    print(f"AI: P0 Auth={base_env.game.players[0].authority} P1 Auth={base_env.game.players[1].authority} | "
                          f"Combat={base_env.game.players[0].combat_pool} Scrap={action[5]} Buy={action[6]}")
                    
                    obs, reward, done, _, _ = env.step(action)
                    if done:
                        continue
                
                # Game loop
                turn = 0
                while not done:
                    turn += 1
                    current_player_idx = base_env.game.current
                    
                    # Take action based on whose turn it is
                    if current_player_idx == rl_player_idx:
                        action, _ = model.predict(obs, deterministic=True)
                        agent_type = "RL"
                    else:
                        current_player = base_env.game.players[current_player_idx]
                        opponent = base_env.game.players[1 - current_player_idx]
                        ai_action = ai.get_action(current_player, opponent)
                        
                        action = np.zeros(7, dtype=np.int64)
                        for idx in ai_action.get("play_cards", []):
                            if idx < 5:
                                action[idx] = 1
                        action[5] = min(ai_action.get("scrap_explorers", 0), 5)
                        action[6] = min(ai_action.get("buy_explorers", 0), 5)
                        agent_type = "AI"
                    
                    # Take step and print summary line
                    p0_auth_before = base_env.game.players[0].authority
                    p1_auth_before = base_env.game.players[1].authority
                    combat_before = base_env.game.players[current_player_idx].combat_pool
                    
                    obs, reward, done, _, _ = env.step(action)
                    
                    p0_auth_after = base_env.game.players[0].authority
                    p1_auth_after = base_env.game.players[1].authority
                    
                    # Calculate damage dealt
                    damage_dealt = 0
                    if current_player_idx == 0:
                        damage_dealt = p1_auth_before - p1_auth_after
                    else:
                        damage_dealt = p0_auth_before - p0_auth_after
                        
                    print(f"{agent_type}: P{current_player_idx} Auth={base_env.game.players[current_player_idx].authority} | "
                          f"Damage={damage_dealt} Scrap={action[5]} Buy={action[6]}")
                
                # Game over summary
                winner = 0 if base_env.game.players[1].authority <= 0 else 1
                rl_won = winner == rl_player_idx
                print(f"Game {i+1} result: RL Agent {'WON' if rl_won else 'LOST'} | "
                      f"Final: P0={base_env.game.players[0].authority} P1={base_env.game.players[1].authority}")

def show_winning_match(model_path, as_player=0, max_attempts=100):
    """Show a terse summary of a single match where the RL agent wins."""
    # Load the trained model
    model = PPO.load(model_path)
    ai = GreedyCombatAI()
    
    print(f"\n=== Looking for a match where RL Agent wins as Player {as_player} ===")
    
    for attempt in range(max_attempts):
        print(f"Attempt {attempt+1}/{max_attempts}...")
        
        # Create fresh environment
        base_env = ExplorerOnlyGymEnv(render_mode=None)
        env = FlattenActionWrapper(base_env)
        obs, _ = env.reset()
        done = False
        
        # If RL is player 1, let AI take first turn
        if as_player == 1:
            current_player = base_env.game.players[0]
            opponent = base_env.game.players[1]
            ai_action = ai.get_action(current_player, opponent)
            
            action = np.zeros(7, dtype=np.int64)
            for idx in ai_action.get("play_cards", []):
                if idx < 5:
                    action[idx] = 1
            action[5] = min(ai_action.get("scrap_explorers", 0), 5)
            action[6] = min(ai_action.get("buy_explorers", 0), 5)
            
            obs, _, done, _, _ = env.step(action)
            if done:
                continue
                
        # Store match history
        match_history = []
        turn = 0
        
        # Play the game
        while not done:
            turn += 1
            current_player_idx = base_env.game.current
            
            # Initial state for this turn
            p0_auth_before = base_env.game.players[0].authority
            p1_auth_before = base_env.game.players[1].authority
            
            # Determine action based on current player
            if current_player_idx == as_player:
                action, _ = model.predict(obs, deterministic=True)
                agent_type = "RL"
            else:
                current_player = base_env.game.players[current_player_idx]
                opponent = base_env.game.players[1 - current_player_idx]
                ai_action = ai.get_action(current_player, opponent)
                
                action = np.zeros(7, dtype=np.int64)
                for idx in ai_action.get("play_cards", []):
                    if idx < 5:
                        action[idx] = 1
                action[5] = min(ai_action.get("scrap_explorers", 0), 5)
                action[6] = min(ai_action.get("buy_explorers", 0), 5)
                agent_type = "AI"
            
            # Take action
            obs, _, done, _, _ = env.step(action)
            
            # Final state after action
            p0_auth_after = base_env.game.players[0].authority
            p1_auth_after = base_env.game.players[1].authority
            
            # Calculate damage
            damage_dealt = 0
            if current_player_idx == 0:
                damage_dealt = p1_auth_before - p1_auth_after
            else:
                damage_dealt = p0_auth_before - p0_auth_after
                
            # Record turn summary
            turn_summary = {
                "turn": turn,
                "agent": agent_type,
                "player": current_player_idx,
                "p0_auth": p0_auth_after,
                "p1_auth": p1_auth_after,
                "damage": damage_dealt,
                "scrap": int(action[5]),
                "buy": int(action[6])
            }
            match_history.append(turn_summary)
        
        # Check if RL agent won
        winner = 0 if base_env.game.players[1].authority <= 0 else 1
        if winner == as_player:
            print(f"\n=== Found winning match on attempt {attempt+1} ===")
            print(f"Match where RL Agent won as Player {as_player}\n")
            
            # Print match summary
            print(f"{'Turn':4} | {'Agent':2} | {'P0':3} | {'P1':3} | {'Dmg':3} | {'Scrp':4} | {'Buy':3}")
            print("-" * 40)
            
            for turn in match_history:
                print(f"{turn['turn']:4} | {turn['agent']:2} | "
                      f"{turn['p0_auth']:3} | {turn['p1_auth']:3} | "
                      f"{turn['damage']:3} | {turn['scrap']:4} | {turn['buy']:3}")
            
            print("-" * 40)
            print(f"Final: P0={base_env.game.players[0].authority}, P1={base_env.game.players[1].authority}")
            print(f"RL Agent won as Player {as_player}!")
            return True
    
    print(f"No winning matches found in {max_attempts} attempts.")
    return False

class FlattenActionWrapper(gym.ActionWrapper):
    """Wrapper that flattens a Dict action space into a MultiDiscrete space."""
    
    def __init__(self, env):
        super().__init__(env)
        self.original_action_space = env.action_space
        
        # Create a MultiDiscrete action space:
        # - 5 binary values for play_cards (0-1 for each)
        # - 1 value for scrap_explorers (0-5)
        # - 1 value for buy_explorers (0-5)
        self.action_space = spaces.MultiDiscrete([2, 2, 2, 2, 2, 6, 6])
    
    def action(self, action):
        """Convert the MultiDiscrete action back to the original Dict format."""
        # Extract parts of the MultiDiscrete action
        play_cards = action[:5].astype(bool)  # First 5 values for play_cards
        scrap_explorers = int(action[5])      # 6th value for scrap_explorers
        buy_explorers = int(action[6])        # 7th value for buy_explorers
        
        # Construct the Dict action
        dict_action = {
            "play_cards": play_cards,
            "scrap_explorers": scrap_explorers,
            "buy_explorers": buy_explorers
        }
        
        return dict_action

if __name__ == "__main__":
    mode = input("Select mode (train/evaluate/benchmark/debug/winning-match): ").strip().lower()
    
    if mode == "train":
        timesteps = int(input("Enter training timesteps (default: 1000000): ") or "1000000")
        continue_training = input("Continue from previous model? (y/n): ").strip().lower() == 'y'
        
        if continue_training:
            model_path = input("Enter model path (default: ../models/final_model): ").strip()
            if not model_path:
                model_path = os.path.join(models_dir, "final_model")
            if not model_path.endswith('.zip'):
                model_path = model_path + '.zip'
            train_agent(total_timesteps=timesteps, continue_from=model_path)
        else:
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
        results = benchmark_against_ai(model_path, episodes_per_ai=episodes)
        
        # Display results summary
        print("\n=== BENCHMARK SUMMARY ===")
        for ai_name, stats in results.items():
            print(f"Results vs {ai_name}:")
            print(f"  As player 0: {stats['as_p0']:.2f}")
            print(f"  As player 1: {stats['as_p1']:.2f}")
            print(f"  Overall: {stats['overall']:.2f}")
    elif mode == "debug":
        model_path = input("Enter model path (e.g., ../models/final_model): ").strip()
        if not model_path:
            model_path = os.path.join(models_dir, "final_model")
        episodes = int(input("Enter games per benchmark AI (default: 2): ") or "2")
        debug_benchmark_against_ai(model_path, episodes_per_ai=episodes)
    elif mode == "winning-match":
        model_path = input("Enter model path (e.g., ../models/final_model): ").strip()
        if not model_path:
            model_path = os.path.join(models_dir, "final_model")
        
        player = int(input("Show RL agent winning as which player? (0/1): ") or "0")
        show_winning_match(model_path, as_player=player)
    else:
        print("Invalid mode selected")