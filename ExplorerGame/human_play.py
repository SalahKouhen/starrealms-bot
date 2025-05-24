import os
from explorer_only_env import ExplorerOnlyGame
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from train_rl_agent import FlattenActionWrapper
from gym_wrapper import ExplorerOnlyGymEnv
import numpy as np

def play_against_rl_agent(model_path, human_player_idx=0):
    """Play a game where the human plays against a trained RL agent."""
    # Load the trained model
    model = PPO.load(model_path)
    
    # Create the game
    gym_env = ExplorerOnlyGymEnv()
    wrapped_env = FlattenActionWrapper(gym_env)
    game = gym_env.game  # Access the underlying game
    
    # Create a vectorized environment for normalization
    vec_env = DummyVecEnv([lambda: wrapped_env])
    
    # Load the saved normalization parameters
    models_dir = "../models"  # Fix the path to make it more reliable
    norm_path = os.path.join(models_dir, "vec_normalize.pkl")
    
    if os.path.exists(norm_path):
        print("Loading normalization parameters...")
        vec_env = VecNormalize.load(norm_path, vec_env)
        # Don't update the normalization parameters
        vec_env.training = False
        vec_env.norm_reward = False
    else:
        print("Warning: Normalization parameters not found.")
        # Create default normalization for key continuous variables
        vec_env = VecNormalize(
            vec_env, 
            norm_obs=True, 
            norm_reward=False,
            norm_obs_keys=[
                "p0_authority", "p0_trade_pool", "p0_combat_pool",
                "p1_authority", "p1_trade_pool", "p1_combat_pool"
            ]
        )
    
    # Reset the environment - handle both possible return formats
    reset_result = vec_env.reset()
    if isinstance(reset_result, tuple):
        # New format returns (obs, info)
        obs = reset_result[0]
    else:
        # Old format returns just obs
        obs = reset_result
    
    # The reset method returns observations as a batch (for all envs in the VecEnv)
    # Extract the first (and only) observation
    if isinstance(obs, np.ndarray) and obs.shape[0] > 0:
        obs = obs[0]
    
    done = False
    
    print("=== Star Realms: Human vs RL Agent ===")
    print(f"You are Player {human_player_idx}")
    
    while not done:
        # Display game state
        game._display_game_state()
        
        # Determine if it's human's turn
        is_human_turn = game.current == human_player_idx
        
        if is_human_turn:
            # Human player's turn
            action_dict = game._get_human_action()
            
            # Convert human action to gym format
            gym_action = gym_env._convert_game_action_to_gym_action(action_dict)
            
            # Convert to flattened format for the wrapper
            flattened_action = np.zeros(7, dtype=np.int64)
            
            # Set play_cards (first 5 elements)
            for idx in action_dict.get("play_cards", []):
                if idx < 5:  # Only consider first 5 cards
                    flattened_action[idx] = 1
            
            # Set scrap_explorers and buy_explorers
            flattened_action[5] = min(action_dict.get("scrap_explorers", 0) 
                                    if isinstance(action_dict.get("scrap_explorers", 0), int)
                                    else len(action_dict.get("scrap_explorers", [])), 5)
            flattened_action[6] = min(action_dict.get("buy_explorers", 0), 5)
            
            action = flattened_action
        else:
            # RL agent's turn
            print(f"\nPlayer {game.current}'s turn (RL Agent)...")
            
            # Get action from RL model with normalized observation
            raw_action, _ = model.predict(obs, deterministic=True)
            print("RL Agent is thinking...")
            
            # Format the action correctly based on what's returned
            if np.isscalar(raw_action) or (isinstance(raw_action, np.ndarray) and raw_action.size == 1):
                print(f"Debug - Action is scalar: {raw_action}")
                # Model is using a Discrete action space - convert to our MultiDiscrete format
                # This likely happens if your model was trained with a different wrapper
                
                # Convert to integer
                discrete_action = int(raw_action)
                
                # Create a properly formatted action array
                action = np.zeros(7, dtype=np.int64)
                
                # Map the discrete action to our format (simplified mapping)
                # This is a basic mapping - you may need to adjust based on your training
                play_idx = discrete_action % 32  # 2^5 combinations for play_cards
                scrap_count = (discrete_action // 32) % 6  # 0-5 for scrap_explorers
                buy_count = (discrete_action // 192) % 6   # 0-5 for buy_explorers
                
                # Set play_cards bits (first 5 positions)
                for i in range(5):
                    action[i] = (play_idx >> i) & 1
                
                # Set scrap and buy counts
                action[5] = scrap_count
                action[6] = buy_count
                
                print(f"Converted to: play={action[:5]}, scrap={action[5]}, buy={action[6]}")
            else:
                # Action is already in expected format
                print(f"Debug - Action shape: {raw_action.shape}, type: {type(raw_action)}")
                action = raw_action
                
                # Ensure correct shape if needed
                if len(action.shape) > 1:
                    action = action.flatten()

        # Take step in vectorized environment (normalizes observation)
        action = action.reshape(-1)  # Ensure it's flattened
        print(f"Final action: {action}")
        next_obs, reward, done, info = vec_env.step(np.array([action]))  # DummyVecEnv expects batch of actions

        # Extract the first (and only) elements from the batched returns
        if isinstance(next_obs, np.ndarray) and next_obs.shape[0] > 0:
            obs = next_obs[0]
        else:
            obs = next_obs
            
        if isinstance(done, np.ndarray) and done.shape[0] > 0:
            done = done[0]
        
        # Report what happened
        if not done:
            print(f"Player {1-game.current} ended their turn.")
    
    # Game over
    winner = 0 if game.players[1].authority <= 0 else 1
    print(f"\n=== Game Over ===")
    print(f"Player {winner} wins!")
    print(f"Final score: Player 0: {game.players[0].authority} authority, " 
          f"Player 1: {game.players[1].authority} authority")
    
    # Determine if human won
    human_won = winner == human_player_idx
    print(f"{'You won!' if human_won else 'RL agent won!'}")
    
    return winner

def play_game():
    # Ask user which mode they want to play
    print("Star Realms Explorer-Only Game")
    print("1. Play against simple AI")
    print("2. Play against trained RL agent")
    print("3. Watch AI-only game")
    
    mode_choice = input("Select mode (1/2/3): ").strip()
    
    if mode_choice == '2':
        # Play against RL agent
        models_dir = "../models"
        default_model = os.path.join(models_dir, "final_model")
        
        # Let user choose a model or use default
        model_path = input(f"Enter model path (default: {default_model}): ").strip()
        if not model_path:
            model_path = default_model
            
        player_choice = input("Which player do you want to be? (0/1): ").strip()
        player_idx = 0 if player_choice != '1' else 1
        
        play_against_rl_agent(model_path, human_player_idx=player_idx)
    elif mode_choice == '1':
        # Play against simple AI
        game = ExplorerOnlyGame()
        player_choice = input("Which player do you want to be? (0/1): ").strip()
        player_idx = 0 if player_choice != '1' else 1
        game.play_interactive_game(human_player_idx=player_idx)
    else:
        # Watch AI-only game
        game = ExplorerOnlyGame()
        winner = game.play_quick_game()
        print(f"Winner: Player {winner}")

if __name__ == "__main__":
    play_game()