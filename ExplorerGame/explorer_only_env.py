import random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

# -----------------------------
# Basic card model
# -----------------------------

@dataclass
class Card:
    """Minimal representation of a Star Realms card for the Explorer‑only variant."""

    name: str
    cost: int
    trade: int = 0
    combat: int = 0
    scrap_combat: int = 0  # Combat gained when this card is scrapped while in play

    def is_explorer(self) -> bool:
        return self.name == "Explorer"

    def copy(self) -> "Card":
        # Cheap deep‑copy because Card is immutable after creation
        return Card(
            self.name, self.cost, self.trade, self.combat, self.scrap_combat
        )


# -----------------------------
# Player state container
# -----------------------------

@dataclass
class Player:
    deck: List[Card] = field(default_factory=list)
    hand: List[Card] = field(default_factory=list)
    discard: List[Card] = field(default_factory=list)
    in_play: List[Card] = field(default_factory=list)
    authority: int = 50  # Starting health
    trade_pool: int = 0
    combat_pool: int = 0

    # ---- convenience helpers ----

    def shuffle_discard_into_deck(self) -> None:
        """Move the entire discard pile into the deck and shuffle in‑place."""
        random.shuffle(self.discard)
        self.deck.extend(self.discard)
        self.discard.clear()

    def draw(self, n: int = 1) -> None:
        """Draw n cards; reshuffle automatically if deck runs out."""
        for _ in range(n):
            if not self.deck:
                self.shuffle_discard_into_deck()
                if not self.deck:  # Empty discard as well ⇒ cannot draw
                    return
            self.hand.append(self.deck.pop())

    # ---- turn helpers ----

    def reset_pools(self) -> None:
        self.trade_pool = 0
        self.combat_pool = 0

    def cleanup_phase(self) -> None:
        """Move all in‑play cards to discard pile and draw a new hand."""
        self.discard.extend(self.in_play)
        self.in_play.clear()
        self.draw(5)


# -----------------------------
# Game engine: Explorer‑only variant
# -----------------------------

class ExplorerOnlyGame:
    """A *deterministic* two‑player Star Realms game with no trade row.

    • Players start with 8 Scouts and 2 Vipers shuffled into a 10‑card deck.
    • The only purchasable card is *Explorer* (cost 2).
    • Explorer may be scrapped *while in play* for +2 combat (discard‑pile scrap is
      ignored in this simple variant).
    • Game ends when a player drops to 0 or fewer authority.

    The class exposes a Gym‑style `reset()` and `step()` API so it can be plugged
    directly into RL libraries.
    """

    # ---- static card definitions ----

    SCOUT = Card("Scout", cost=0, trade=1)
    VIPER = Card("Viper", cost=0, combat=1)
    EXPLORER = Card("Explorer", cost=2, trade=2, scrap_combat=2)

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
        self.players: List[Player] = [Player(), Player()]
        self.current: int = 0  # Index of the player whose turn it is
        self.turn: int = 0
        self._init_players()

    # ------------------------------------------------------------------
    # Public API (Gym‑like)
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> Dict:
        if seed is not None:
            random.seed(seed)
        for p in self.players:
            p.deck.clear(); p.hand.clear(); p.discard.clear(); p.in_play.clear()
            p.authority = 50
        self._init_players()
        self.current = 0
        self.turn = 0
        return self._get_observation()

    def step(self, action: Dict) -> Tuple[Dict, int, bool, Dict]:
        """Advance the game by one *full turn* for the current player."""
        player = self.players[self.current]

        # 1️⃣ ACTION PHASE – play selected cards
        player.reset_pools()
        
        # Play selected cards from hand
        if "play_cards" in action:
            # Get indices to play, sorted in reverse so we don't disturb remaining indices
            indices = sorted(action["play_cards"], reverse=True)
            
            # Validate indices
            valid_indices = [i for i in indices if 0 <= i < len(player.hand)]
            
            # Play each card
            for idx in valid_indices:
                card = player.hand.pop(idx)
                player.in_play.append(card)
                player.trade_pool += card.trade
                player.combat_pool += card.combat

        # 2️⃣  SCRAP PHASE (optional)
        if "scrap_explorers" in action and action["scrap_explorers"]:
            # Get all explorers in play with their indices
            explorers_indices = [i for i, c in enumerate(player.in_play) if c.is_explorer()]
            
            # If action["scrap_explorers"] is a list, scrap specific explorers
            if isinstance(action["scrap_explorers"], list):
                # Important fix: Need to sort in reverse to avoid index shifting issues
                indices_to_scrap = sorted(
                    [i for i in action["scrap_explorers"] if i in explorers_indices], 
                    reverse=True
                )
            else:
                # For backward compatibility: if boolean True, scrap all explorers
                indices_to_scrap = sorted(explorers_indices, reverse=True)
                
            # Debug output for combat tracking
            combat_before = player.combat_pool
                
            # Scrap the selected explorers
            for idx in indices_to_scrap:
                if idx < len(player.in_play) and player.in_play[idx].is_explorer():
                    card = player.in_play.pop(idx)
                    player.combat_pool += card.scrap_combat
                    player.discard.append(card)  # Scrapped cards go to discard
            
            # Debug output for combat tracking
            combat_after = player.combat_pool
            print(f"DEBUG: Combat from scrapping: {combat_after - combat_before}")

        # 3️⃣  BUY PHASE – only Explorers are available
        to_buy = min(action.get("buy_explorers", 0), player.trade_pool // 2)
        if to_buy > 0:
            print(f"DEBUG: Buying {to_buy} Explorer(s)")
            for _ in range(to_buy):
                player.trade_pool -= 2
                player.discard.append(self.EXPLORER.copy())
                
            # Debug info about player's cards
            explorer_count = sum(1 for c in player.discard if c.is_explorer())
            print(f"DEBUG: Player now has {explorer_count} Explorer(s) in discard pile")

        # 4️⃣  COMBAT PHASE – damage the opponent
        opponent = self.players[1 - self.current]
        print(f"DEBUG: Final combat pool: {player.combat_pool}, opponent authority before: {opponent.authority}")
        opponent.authority -= player.combat_pool
        print(f"DEBUG: Opponent authority after: {opponent.authority}")

        # 5️⃣  CLEANUP PHASE
        player.discard.extend(player.in_play)
        player.discard.extend(player.hand)  # Discard any unplayed cards
        player.in_play.clear()
        player.hand.clear()
        player.draw(5)  # Everyone draws 5 after first turn

        # 6️⃣  Check end of game
        done = opponent.authority <= 0 or player.authority <= 0
        reward = 0
        if done:
            reward = 1 if opponent.authority <= 0 else -1

        # 7️⃣  Next player's turn
        self.current = 1 - self.current
        self.turn += 1
        return self._get_observation(), reward, done, {}

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _init_players(self) -> None:
        """Give each player a fresh 10-card deck with appropriate starting hands."""
        for i, p in enumerate(self.players):
            p.deck = (
                [self.SCOUT.copy() for _ in range(8)]
                + [self.VIPER.copy() for _ in range(2)]
            )
            random.shuffle(p.deck)
            # First player draws 3, second player draws 5
            p.draw(3 if i == 0 else 5)

    def _get_observation(self) -> Dict:
        """Return a *fully observable* dict; easy to mask later for RL."""
        return {
            "current_player": self.current,
            "turn": self.turn,
            "players": [
                {
                    "authority": p.authority,
                    "hand_size": len(p.hand),
                    "deck_size": len(p.deck),
                    "discard_size": len(p.discard),
                    "in_play_names": [c.name for c in p.in_play],
                }
                for p in self.players
            ],
        }

    # ------------------------------------------------------------------
    # Convenience: quick self‑play until end of game (no RL) --------------
    # ------------------------------------------------------------------

    def play_quick_game(self, scrap_chance: float = 0.5) -> int:
        """Run a game with random decisions; returns winner index (0/1)."""
        obs = self.reset()
        done = False
        while not done:
            # Simple random policy: play all cards, buy as many Explorers as possible, scrap randomly
            current_player = self.players[self.current]
            action = {
                "play_cards": list(range(len(current_player.hand))),  # Play all cards
                "scrap_explorers": random.random() < scrap_chance,
                "buy_explorers": 10,  # high upper bound; env will cap by trade
            }
            obs, reward, done, _ = self.step(action)
        return 0 if reward == 1 else 1

    # ------------------------------------------------------------------
    # Interactive human play mode
    # ------------------------------------------------------------------

    def play_interactive_game(self, human_player_idx: int = 0) -> int:
        """Play a game where one player is human-controlled."""
        obs = self.reset()
        done = False
        
        print("=== Star Realms Explorer-Only Game ===")
        print(f"You are Player {human_player_idx}")
        
        while not done:
            # Display game state
            self._display_game_state()
            
            # Determine if it's human's turn
            is_human_turn = self.current == human_player_idx
            
            if is_human_turn:
                # Human player's turn
                action = self._get_human_action()
            else:
                # AI player's turn (simple strategy)
                print(f"\nPlayer {self.current}'s turn (AI)...")
                current_ai_player = self.players[self.current]
                
                # AI plays all cards
                action = {
                    "play_cards": list(range(len(current_ai_player.hand))),
                }
                
                # AI decides whether to scrap explorers
                explorers_in_play = [(i, c) for i, c in enumerate(current_ai_player.in_play) if c.is_explorer()]
                
                # Simple AI strategy: scrap explorers if it has enough trade to buy at least one new one
                # or if opponent is below 10 authority
                opponent = self.players[1 - self.current]
                needs_combat = opponent.authority < 10
                
                if explorers_in_play:
                    if needs_combat:
                        # Need combat to finish opponent - scrap all
                        action["scrap_explorers"] = [idx for idx, _ in explorers_in_play]
                    else:
                        # Scrap randomly between 0 and all explorers
                        num_to_scrap = random.randint(0, len(explorers_in_play))
                        indices_to_scrap = random.sample([idx for idx, _ in explorers_in_play], num_to_scrap)
                        action["scrap_explorers"] = indices_to_scrap
                
                # AI always tries to buy as many Explorers as possible
                action["buy_explorers"] = 10  # Will be capped by available trade
                
                # Show what AI is doing
                print(f"AI plays {len(current_ai_player.hand)} cards")
                if action.get("scrap_explorers"):
                    print(f"AI scraps {len(action['scrap_explorers'])} Explorer(s)")
            
            # Take the action
            obs, reward, done, _ = self.step(action)
            
            # Report what happened
            if not done:
                print(f"Player {1-self.current} ended their turn.")
            
        # Game over
        winner = 0 if self.players[1].authority <= 0 else 1
        print(f"\n=== Game Over ===")
        print(f"Player {winner} wins!")
        print(f"Final score: Player 0: {self.players[0].authority} authority, " 
              f"Player 1: {self.players[1].authority} authority")
        
        return winner
    
    def _display_game_state(self) -> None:
        """Display the current game state in a readable format."""
        current = self.players[self.current]
        opponent = self.players[1 - self.current]
        
        print("\n" + "="*50)
        print(f"Turn {self.turn}: Player {self.current}'s turn")
        print("="*50)
        
        # Show player stats
        print(f"Player {self.current}: {current.authority} authority | Player {1-self.current}: {opponent.authority} authority")
        
        # Show cards in hand
        print("\nYour hand:")
        for i, card in enumerate(current.hand):
            print(f"  [{i}] {card.name} (Trade: {card.trade}, Combat: {card.combat})")
        
        # Show cards in play
        if current.in_play:
            print("\nCards in play:")
            for i, card in enumerate(current.in_play):
                scrap_text = f", Scrap: +{card.scrap_combat} combat" if card.scrap_combat > 0 else ""
                print(f"  {i+1}. {card.name} (Trade: {card.trade}, Combat: {card.combat}{scrap_text})")
        
        # Show current pools
        print(f"\nTrade pool: {current.trade_pool}")
        print(f"Combat pool: {current.combat_pool}")
        
        # Show deck/discard info
        print(f"\nDeck: {len(current.deck)} cards | Discard: {len(current.discard)} cards")
        
    def _get_human_action(self) -> Dict:
        """Get action input from the human player."""
        action = {}
        current = self.players[self.current]
        
        # Step 1: Play cards from hand
        if current.hand:
            print("\n=== PLAY PHASE ===")
            print("Select cards to play from your hand:")
            print("Type 'a' to play your entire hand at once")
            
            # Create a working copy of the hand to simulate removals
            temp_hand = current.hand.copy()
            played_indices = []
            temp_trade = 0
            temp_combat = 0
            
            # Create a temporary list to track what would be in play
            temp_in_play = current.in_play.copy()
            # Store which cards would be played this turn (for scrapping decision)
            newly_played_cards = []
            
            while temp_hand:
                # Display remaining cards
                for i, card in enumerate(temp_hand):
                    print(f"  [{i}] {card.name} (Trade: {card.trade}, Combat: {card.combat})")
                
                choice = input("\nEnter card number to play, 'a' to play everything, or press Enter to finish: ").strip().lower()
                
                if not choice:  # Empty input means done playing
                    break
                    
                if choice == 'a':
                    # Play all remaining cards at once
                    for i, card in enumerate(temp_hand):
                        # Find each card's index in the original hand
                        for j, original_card in enumerate(current.hand):
                            if j not in played_indices and original_card.name == card.name:
                                played_indices.append(j)
                                # Add to simulated in_play area
                                temp_in_play.append(card)
                                newly_played_cards.append(card)
                                break
                        
                        # Update pool totals
                        temp_trade += card.trade
                        temp_combat += card.combat
                    
                    print(f"Playing all cards (+{temp_trade} trade, +{temp_combat} combat)")
                    break  # Exit the loop since we've played everything
                    
                try:
                    idx = int(choice)
                    if 0 <= idx < len(temp_hand):
                        # Get the actual card and its index in the original hand
                        card = temp_hand.pop(idx)
                        
                        # Find where this card is in the original hand
                        for i, original_card in enumerate(current.hand):
                            if i not in played_indices and original_card.name == card.name:
                                played_indices.append(i)
                                # Add to simulated in_play area
                                temp_in_play.append(card)
                                newly_played_cards.append(card)
                                break
                        
                        # Update pool totals
                        temp_trade += card.trade
                        temp_combat += card.combat
                        
                        print(f"Selected {card.name} to play (+{card.trade} trade, +{card.combat} combat)")
                        print(f"Current pools: Trade: {temp_trade}, Combat: {temp_combat}")
                    else:
                        print("Invalid card number. Try again.")
                except ValueError:
                    if choice != '':  # Only show error if they entered something
                        print("Please enter a valid number, 'a', or press Enter.")
            
            action["play_cards"] = played_indices
            
            print(f"\nAfter playing: Trade pool will be {temp_trade}, Combat pool will be {temp_combat}")
        else:
            action["play_cards"] = []
            temp_trade = 0
            temp_combat = 0
            temp_in_play = current.in_play.copy()
        
        # Step 2: Scrap explorers - NOW INCLUDES NEWLY PLAYED EXPLORERS
        all_explorers_in_play = [(i, c) for i, c in enumerate(temp_in_play) if c.is_explorer()]
        
        if all_explorers_in_play:
            print("\n=== SCRAP PHASE ===")
            print(f"You have {len(all_explorers_in_play)} Explorer(s) in play (including ones you just played).")
            
            print("Select Explorers to scrap (each gives +2 combat):")
            print("Explorers in play:")
            for idx, card in all_explorers_in_play:
                # Note whether this was just played this turn
                just_played = card in newly_played_cards if 'newly_played_cards' in locals() else False
                print(f"  [{idx}] {card.name}{' (just played)' if just_played else ''}")
            
            print("\nOptions:")
            print("  [a] Scrap ALL Explorers")
            print("  [n] Scrap NO Explorers")
            print("  Or enter specific indices separated by spaces (e.g., '0 2')")
            
            choice = input("\nYour choice: ").strip().lower()
            
            if choice == 'a':
                # Scrap all explorers
                action["scrap_explorers"] = [idx for idx, _ in all_explorers_in_play]  # FIXED: Extract just the index
                print(f"Scrapping all {len(all_explorers_in_play)} Explorers for +{len(all_explorers_in_play) * 2} combat")
                # Update the combat total for the buy phase
                temp_combat += len(all_explorers_in_play) * 2
            elif choice == 'n' or not choice:
                # Scrap no explorers
                action["scrap_explorers"] = []
                print("Not scrapping any Explorers.")
            else:
                # Parse indices
                try:
                    indices = [int(idx) for idx in choice.split()]
                    valid_indices = [idx for idx in indices if any(idx == i for i, _ in all_explorers_in_play)]
                    
                    if valid_indices:
                        action["scrap_explorers"] = valid_indices
                        print(f"Scrapping {len(valid_indices)} Explorer(s) for +{len(valid_indices) * 2} combat")
                        # Update the combat total for the buy phase
                        temp_combat += len(valid_indices) * 2
                    else:
                        print("No valid Explorer indices. Not scrapping any.")
                        action["scrap_explorers"] = []
                except ValueError:
                    print("Invalid input. Not scrapping any Explorers.")
                    action["scrap_explorers"] = []
        else:
            action["scrap_explorers"] = []
        
        # Step 3: Buy explorers
        print("\n=== BUY PHASE ===")
        max_explorers_to_buy = temp_trade // 2
        
        if max_explorers_to_buy > 0:
            print(f"You can buy up to {max_explorers_to_buy} Explorer(s) (2 trade each).")
            print("Type 'a' to buy the maximum number possible.")
            
            choice = input(f"How many Explorers to buy (0-{max_explorers_to_buy})? ").strip().lower()
            
            if choice == 'a':
                # Buy maximum possible explorers
                action["buy_explorers"] = max_explorers_to_buy
                print(f"Buying maximum: {max_explorers_to_buy} Explorer(s)")
            else:
                try:
                    num = int(choice)
                    if 0 <= num <= max_explorers_to_buy:
                        action["buy_explorers"] = num
                    else:
                        print("Invalid number. Buying 0.")
                        action["buy_explorers"] = 0
                except ValueError:
                    print("Invalid input. Buying 0.")
                    action["buy_explorers"] = 0
        else:
            print("You don't have enough trade to buy Explorers.")
            action["buy_explorers"] = 0
        
        # Add a turn summary to help the player understand what will happen
        print("\n=== TURN SUMMARY ===")
        print(f"• Playing {len(action['play_cards'])} cards")
        if "scrap_explorers" in action and action["scrap_explorers"]:
            print(f"• Scrapping {len(action['scrap_explorers'])} Explorer(s) for +{len(action['scrap_explorers']) * 2} combat")
        print(f"• Buying {action.get('buy_explorers', 0)} Explorer(s)")
        print(f"• Final pools: Trade {temp_trade - (action.get('buy_explorers', 0) * 2)}, Combat {temp_combat}")
        print(f"• Dealing {temp_combat} damage to opponent")
        
        # Confirm end of turn
        input("\nPress Enter to end your turn...")
        
        return action


# -----------------------------
# Quick demo when run as script
# -----------------------------

if __name__ == "__main__":
    game = ExplorerOnlyGame()
    
    # Ask user if they want to play or run an AI-only game
    play_choice = input("Do you want to play the game? (y/n): ").strip().lower()
    
    if play_choice == 'y':
        player_choice = input("Which player do you want to be? (0/1): ").strip()
        player_idx = 0 if player_choice != '1' else 1
        game.play_interactive_game(human_player_idx=player_idx)
    else:
        winner = game.play_quick_game()
        print(f"Winner: Player {winner}")
