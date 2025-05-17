from explorer_only_env import ExplorerOnlyGame

def play_game():
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

if __name__ == "__main__":
    play_game()