from typing import Dict, List, Optional
import random
from explorer_only_env import Player

class BenchmarkAI:
    """Base class for all benchmark AI strategies."""

    def __init__(self, name: str):
        self.name = name

    def get_action(self, player: Player, opponent: Player) -> Dict:
        """Get action for the current player state."""
        raise NotImplementedError("Subclasses must implement this method")


class RandomAI(BenchmarkAI):
    """Makes completely random legal moves."""

    def __init__(self):
        super().__init__("Random")

    def get_action(self, player: Player, opponent: Player) -> Dict:
        # Play random subset of cards
        play_indices = random.sample(
            range(len(player.hand)),
            random.randint(0, len(player.hand))
        )

        # Randomly decide how many explorers to scrap
        explorers_in_play = sum(1 for card in player.in_play if card.is_explorer())
        scrap_count = random.randint(0, explorers_in_play)
        
        # Random number of explorers to buy (limited by trade)
        max_explorers = player.trade_pool // 2
        buy_count = random.randint(0, max_explorers)

        return {
            "play_cards": play_indices,
            "scrap_explorers": scrap_count,  # Changed from scrap_explorers_count
            "buy_explorers": buy_count
        }


class GreedyTradeAI(BenchmarkAI):
    """Focuses on building economy by buying explorers. Scraps when no Scouts in hand."""

    def __init__(self):
        super().__init__("Greedy Trade")

    def get_action(self, player: Player, opponent: Player) -> Dict:
        # Always play all cards
        play_indices = list(range(len(player.hand)))
        
        # Check if there are any Scouts in hand
        has_scouts_in_hand = any(card.name == "Scout" for card in player.hand)
        
        # Get count of explorers currently in play
        explorers_in_play = sum(1 for card in player.in_play if card.is_explorer())
        # Count explorers in hand that will be played
        explorers_in_hand = sum(1 for card in player.hand if card.is_explorer())
        
        # If no Scouts in hand, scrap all explorers (switch to combat mode)
        if not has_scouts_in_hand:
            scrap_count = explorers_in_play + explorers_in_hand
        else:
            scrap_count = 0
        
        # Buy as many explorers as possible
        buy_count = player.trade_pool // 2
        
        return {
            "play_cards": play_indices,
            "scrap_explorers": scrap_count,  # Changed from scrap_explorers_count
            "buy_explorers": buy_count
        }


class GreedyCombatAI(BenchmarkAI):
    """Always focuses on dealing damage by scrapping explorers."""

    def __init__(self):
        super().__init__("Greedy Combat")

    def get_action(self, player: Player, opponent: Player) -> Dict:
        # Always play all cards
        play_indices = list(range(len(player.hand)))
        
        # Count explorers in play + hand that will be played
        explorers_in_play = sum(1 for card in player.in_play if card.is_explorer())
        explorers_in_hand = sum(1 for card in player.hand if card.is_explorer())
        
        # Always scrap all explorers for combat
        scrap_count = explorers_in_play + explorers_in_hand
        
        # Buy as many explorers as possible (to scrap next turn)
        buy_count = player.trade_pool // 2
        
        return {
            "play_cards": play_indices,
            "scrap_explorers": scrap_count,  # Changed from scrap_explorers_count
            "buy_explorers": buy_count
        }


class BalancedAI(BenchmarkAI):
    """Uses a balanced approach with simple heuristics."""

    def __init__(self):
        super().__init__("Balanced")

    def get_action(self, player: Player, opponent: Player) -> Dict:
        # Always play all cards
        play_indices = list(range(len(player.hand)))
        
        # Count explorers in play and in hand
        explorers_in_play = sum(1 for card in player.in_play if card.is_explorer())
        explorers_in_hand = sum(1 for card in player.hand if card.is_explorer())
        total_explorers_after_play = explorers_in_play + explorers_in_hand
        
        # Decide how many explorers to scrap based on game state
        if opponent.authority <= total_explorers_after_play * 2:
            # If opponent is low on health, scrap all for the kill
            scrap_count = total_explorers_after_play
        elif total_explorers_after_play >= 3:
            # If we have many explorers, scrap some for balance
            scrap_count = total_explorers_after_play // 2
        elif total_explorers_after_play > 1 and player.trade_pool >= 4:
            # Early game with surplus, just scrap one
            scrap_count = 1
        else:
            scrap_count = 0
        
        # Buy decision based on economy
        max_explorers = player.trade_pool // 2
        
        # Count total explorers in all zones
        deck_explorers = sum(1 for c in player.deck if c.is_explorer())
        discard_explorers = sum(1 for c in player.discard if c.is_explorer())
        total_explorers = deck_explorers + discard_explorers + explorers_in_hand + explorers_in_play
        
        # Buy based on total explorer count
        if total_explorers < 3:
            buy_count = max_explorers  # Fixed: was maxs_explorers
        else:
            buy_count = min(max_explorers, max(0, 5 - total_explorers))
        
        return {
            "play_cards": play_indices,
            "scrap_explorers": scrap_count,  # Changed from scrap_explorers_count
            "buy_explorers": buy_count
        }