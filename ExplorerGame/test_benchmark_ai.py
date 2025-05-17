import random
import os
import sys
from explorer_only_env import ExplorerOnlyGame
from benchmark_ai import RandomAI, GreedyTradeAI, GreedyCombatAI, BalancedAI

def ai_vs_ai(ai1, ai2, games=100, display=False, quiet=False):
    """Run AI vs AI games and report win rates."""
    wins_ai1 = 0
    wins_ai2 = 0
    
    # Save original stdout if we're going quiet
    if quiet:
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    
    for i in range(games):
        # Set different random seed for each game
        seed = random.randint(0, 10000)
        game = ExplorerOnlyGame(seed=seed)
        
        # Set debug flag on game to reduce output
        game.debug_output = not quiet
        
        done = False
        while not done:
            # Get current player
            current_player = game.players[game.current]
            opponent = game.players[1 - game.current]
            
            # Determine which AI to use
            ai = ai1 if game.current == 0 else ai2
            
            # Get action from AI
            action = ai.get_action(current_player, opponent)
            
            # Take step in game
            _, reward, done, _ = game.step(action)
            
            if display and i == 0 and not quiet:  # Only display first game
                game._display_game_state()
        
        # Determine winner
        if game.players[0].authority <= 0:
            wins_ai2 += 1
        elif game.players[1].authority <= 0:
            wins_ai1 += 1
        
        # Only show progress updates if not in quiet mode
        if (i+1) % 10 == 0 and not quiet:
            print(f"Played {i+1}/{games} games")
    
    # Restore stdout if we went quiet
    if quiet:
        sys.stdout = original_stdout
    
    # Print results with clear formatting
    print(f"\n{'='*50}")
    print(f"🏆 RESULTS: {ai1.name} vs {ai2.name} 🏆")
    print(f"{'='*50}")
    print(f"{ai1.name} wins: {wins_ai1}/{games} ({wins_ai1/games:.2f})")
    print(f"{ai2.name} wins: {wins_ai2}/{games} ({wins_ai2/games:.2f})")
    draws = games - wins_ai1 - wins_ai2
    print(f"Draws: {draws}/{games} ({draws/games:.2f})")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    # Tournament of AIs
    ais = [
        RandomAI(),
        GreedyTradeAI(), 
        GreedyCombatAI(),
        BalancedAI()
    ]
    
    display_games = input("Display first game of each matchup? (y/n): ").lower() == 'y'
    games_per_match = int(input("Games per matchup (default: 100): ") or "100")
    quiet_mode = input("Run in quiet mode (suppress debug output)? (y/n): ").lower() == 'y'
    
    print("\n🏆 TOURNAMENT OF AIs 🏆")
    
    # Run all pairs of AIs against each other
    for i in range(len(ais)):
        for j in range(i+1, len(ais)):
            ai_vs_ai(ais[i], ais[j], games=games_per_match, display=display_games, quiet=quiet_mode)