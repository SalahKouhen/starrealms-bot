import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any, Optional

from explorer_only_env import ExplorerOnlyGame, Player, Card

class ExplorerOnlyGymEnv(gym.Env):
    """Gym wrapper for the Explorer-only Star Realms game."""
    
    metadata = {"render_modes": ["human", "ansi"]}
    
    def __init__(self, render_mode=None):
        super().__init__()
        self.game = ExplorerOnlyGame()
        self.render_mode = render_mode
        
        # Define configurable limits for observation space
        self.limits = {
            # Game state limits
            "max_players": 2,
            "max_turns": 100,
            
            # Authority limits
            "min_authority": 0,
            "max_authority": 100,
            
            # Card count limits
            "max_hand_size": 20,
            "max_deck_size": 70,
            "max_discard_size": 70,
            
            # Card type limits
            "max_scout_count": 10,
            "max_viper_count": 10,
            "max_explorer_count": 10,
            
            # Resource limits
            "min_resources": 0,
            "max_trade": 50,
            "max_combat": 50,
            
            # Action space limits
            "max_cards_in_hand": 5,
            "max_explorers_in_play": 5,
            "max_explorers_to_buy": 6
        }
        
        # Define observation space using the configurable limits
        self.observation_space = spaces.Dict({
            # Game state
            "current_player": spaces.Discrete(self.limits["max_players"]),
            "turn": spaces.Discrete(self.limits["max_turns"]),
            
            # Player 0 info
            "p0_authority": spaces.Box(
                low=self.limits["min_authority"], 
                high=self.limits["max_authority"], 
                shape=(1,), 
                dtype=np.float32
            ),
            "p0_hand_size": spaces.Discrete(self.limits["max_hand_size"]),
            "p0_deck_size": spaces.Discrete(self.limits["max_deck_size"]),
            "p0_discard_size": spaces.Discrete(self.limits["max_discard_size"]),
            "p0_scout_in_hand": spaces.Discrete(self.limits["max_scout_count"]),
            "p0_viper_in_hand": spaces.Discrete(self.limits["max_viper_count"]),
            "p0_explorer_in_hand": spaces.Discrete(self.limits["max_explorer_count"]),
            "p0_scout_in_play": spaces.Discrete(self.limits["max_scout_count"]),
            "p0_viper_in_play": spaces.Discrete(self.limits["max_viper_count"]),
            "p0_explorer_in_play": spaces.Discrete(self.limits["max_explorer_count"]),
            "p0_trade_pool": spaces.Box(
                low=self.limits["min_resources"], 
                high=self.limits["max_trade"], 
                shape=(1,), 
                dtype=np.float32
            ),
            "p0_combat_pool": spaces.Box(
                low=self.limits["min_resources"], 
                high=self.limits["max_combat"], 
                shape=(1,), 
                dtype=np.float32
            ),
            
            # Player 1 info (same structure)
            "p1_authority": spaces.Box(
                low=self.limits["min_authority"], 
                high=self.limits["max_authority"], 
                shape=(1,), 
                dtype=np.float32
            ),
            "p1_hand_size": spaces.Discrete(self.limits["max_hand_size"]),
            "p1_deck_size": spaces.Discrete(self.limits["max_deck_size"]),
            "p1_discard_size": spaces.Discrete(self.limits["max_discard_size"]),
            "p1_scout_in_hand": spaces.Discrete(self.limits["max_scout_count"]),
            "p1_viper_in_hand": spaces.Discrete(self.limits["max_viper_count"]),
            "p1_explorer_in_hand": spaces.Discrete(self.limits["max_explorer_count"]),
            "p1_scout_in_play": spaces.Discrete(self.limits["max_scout_count"]),
            "p1_viper_in_play": spaces.Discrete(self.limits["max_viper_count"]),
            "p1_explorer_in_play": spaces.Discrete(self.limits["max_explorer_count"]),
            "p1_trade_pool": spaces.Box(
                low=self.limits["min_resources"], 
                high=self.limits["max_trade"], 
                shape=(1,), 
                dtype=np.float32
            ),
            "p1_combat_pool": spaces.Box(
                low=self.limits["min_resources"], 
                high=self.limits["max_combat"], 
                shape=(1,), 
                dtype=np.float32
            ),
        })
        
        # Define action space using the configurable limits
        self.action_space = spaces.Dict({
            "play_cards": spaces.MultiBinary(self.limits["max_cards_in_hand"]),
            "scrap_explorers": spaces.Discrete(self.limits["max_explorers_in_play"] + 1),  # 0 to max number
            "buy_explorers": spaces.Discrete(self.limits["max_explorers_to_buy"])
        })
    
    def reset(self, seed=None, options=None) -> Tuple[Dict, Dict]:
        """Reset the environment to start a new game."""
        game_obs = self.game.reset(seed=seed)
        observation = self._convert_game_obs_to_gym_obs(game_obs)
        info = {}
        return observation, info
    
    def step(self, action: Dict) -> Tuple[Dict, float, bool, bool, Dict]:
        """Take a step in the environment using the provided action."""
        # Convert gym action format to game action format
        game_action = self._convert_gym_action_to_game_action(action)
        
        # Take the action in the game
        game_obs, reward, done, info = self.game.step(game_action)
        
        # Convert game observation to gym observation
        observation = self._convert_game_obs_to_gym_obs(game_obs)
        
        # In Gymnasium, step returns 5 values including truncated
        truncated = False  # We don't truncate episodes in this game
        
        return observation, reward, done, truncated, info
    
    def render(self):
        """Render the current state of the environment."""
        if self.render_mode == "human" or self.render_mode == "ansi":
            # Use the existing display_game_state method
            self.game._display_game_state()
    
    def _convert_game_obs_to_gym_obs(self, game_obs: Dict) -> Dict:
        """Convert the game observation to the gym observation format with clipping."""
        gym_obs = {}
        
        # Game state - clip turn to max limit
        gym_obs["current_player"] = min(game_obs["current_player"], 1)  # 0 or 1
        gym_obs["turn"] = min(game_obs["turn"], self.limits["max_turns"] - 1)  # Clip to max turns
        
        # Process each player's information with clipping
        for i in range(2):
            p = self.game.players[i]
            prefix = f"p{i}_"
            
            # Authority with clipping (prevent negative values)
            gym_obs[f"{prefix}authority"] = np.clip(
                np.array([p.authority], dtype=np.float32),
                self.limits["min_authority"],
                self.limits["max_authority"]
            )
            
            # Card counts with clipping
            gym_obs[f"{prefix}hand_size"] = min(len(p.hand), self.limits["max_hand_size"] - 1)
            gym_obs[f"{prefix}deck_size"] = min(len(p.deck), self.limits["max_deck_size"] - 1)
            gym_obs[f"{prefix}discard_size"] = min(len(p.discard), self.limits["max_discard_size"] - 1)
            
            # Card types with clipping
            # We need card type limits to define the observation space dimensions
            gym_obs[f"{prefix}scout_in_hand"] = min(
                sum(1 for c in p.hand if c.name == "Scout"),
                self.limits["max_scout_count"] - 1
            )
            gym_obs[f"{prefix}viper_in_hand"] = min(
                sum(1 for c in p.hand if c.name == "Viper"),
                self.limits["max_viper_count"] - 1
            )
            gym_obs[f"{prefix}explorer_in_hand"] = min(
                sum(1 for c in p.hand if c.name == "Explorer"),
                self.limits["max_explorer_count"] - 1
            )
            
            # Card types in play with clipping
            gym_obs[f"{prefix}scout_in_play"] = min(
                sum(1 for c in p.in_play if c.name == "Scout"),
                self.limits["max_scout_count"] - 1
            )
            gym_obs[f"{prefix}viper_in_play"] = min(
                sum(1 for c in p.in_play if c.name == "Viper"),
                self.limits["max_viper_count"] - 1
            )
            gym_obs[f"{prefix}explorer_in_play"] = min(
                sum(1 for c in p.in_play if c.name == "Explorer"),
                self.limits["max_explorer_count"] - 1
            )
            
            # Resource pools with clipping
            gym_obs[f"{prefix}trade_pool"] = np.clip(
                np.array([p.trade_pool], dtype=np.float32),
                self.limits["min_resources"],
                self.limits["max_trade"]
            )
            gym_obs[f"{prefix}combat_pool"] = np.clip(
                np.array([p.combat_pool], dtype=np.float32),
                self.limits["min_resources"],
                self.limits["max_combat"]
            )
        
        return gym_obs
    
    def _convert_gym_action_to_game_action(self, gym_action: Dict) -> Dict:
        """Convert gym action format to the format expected by the game."""
        current_player = self.game.players[self.game.current]
        
        game_action = {}
        
        # Convert play_cards from binary array to list of indices
        if "play_cards" in gym_action:
            # Only consider valid indices (up to hand size)
            valid_size = min(len(current_player.hand), len(gym_action["play_cards"]))
            game_action["play_cards"] = [i for i in range(valid_size) if gym_action["play_cards"][i] == 1]
        
        # In _convert_gym_action_to_game_action method:
        if "scrap_explorers" in gym_action:
            # Just pass through the number of explorers to scrap
            # The environment will handle selecting which ones
            game_action["scrap_explorers"] = int(gym_action["scrap_explorers"])
        
        # Convert buy_explorers from gym format (integer) to game format
        if "buy_explorers" in gym_action:
            game_action["buy_explorers"] = int(gym_action["buy_explorers"])
        
        return game_action