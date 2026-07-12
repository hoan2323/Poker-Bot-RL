"""
play_human.py - Human vs AI Interface
Terminal-based interface for playing poker against trained bot
"""

import os
import sys


def clear_screen():
    """Clear terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')


def card_to_string(card):
    """Convert card index to string"""
    ranks = ['10', 'J', 'Q', 'K', 'A']
    suits = ['C', 'D', 'H', 'S']
    rank = card // 4
    suit = card % 4
    return ranks[rank] + suits[suit]


def card_to_string_hidden(card):
    """Convert card to hidden string"""
    return "??"


def print_game_state(env, player, hide_opponent_cards=True):
    """Print current game state"""
    round_names = ['Preflop', 'Flop', 'Turn', 'River', 'Showdown']
    action_names = {0: 'Check/Call', 1: 'Bet/Raise', 2: 'Fold'}

    print("\n" + "=" * 50)
    print(f"POT: {env.pot} | ROUND: {round_names[env.round]}")
    print("=" * 50)

    # Board
    if env.board:
        board_str = " ".join([card_to_string(c) for c in env.board])
        print(f"BOARD: {board_str}")
    else:
        print("BOARD: ---")

    print("-" * 50)

    # Player hands
    player_hand = env.hands[player]
    opponent_hand = env.hands[1 - player]

    if player_hand:
        player_cards = " ".join([card_to_string(c) for c in player_hand])
        print(f"YOUR HAND: {player_cards}")
    else:
        print("YOUR HAND: ---")

    if hide_opponent_cards and opponent_hand:
        opponent_cards = " ".join([card_to_string_hidden(c) for c in opponent_hand])
        print(f"OPPONENT: {opponent_cards}")
    else:
        if opponent_hand:
            opponent_cards = " ".join([card_to_string(c) for c in opponent_hand])
            print(f"OPPONENT: {opponent_cards}")
        else:
            print("OPPONENT: ---")

    print("-" * 50)

    # Valid actions
    valid_actions = env.get_valid_actions()
    print("VALID ACTIONS:")
    for a in valid_actions:
        print(f"  [{a}] {action_names[a]}")

    # Current player
    if env.current_player == player:
        print("\n>>> YOUR TURN <<<")
    else:
        print("\n>>> OPPONENT'S TURN <<<")


def get_human_action(valid_actions):
    """Get action from human player"""
    print()
    while True:
        try:
            choice = input("Your action: ").strip()
            action = int(choice)
            if action in valid_actions:
                return action
            print(f"Invalid action. Choose from: {valid_actions}")
        except ValueError:
            print("Please enter a number.")


def play_hand(ai_player, human_player=0, verbose=True):
    """Play one hand"""
    # Start game
    state = ai_player.start_new_game(starting_player=human_player)

    if verbose:
        clear_screen()
        print("\n" + "=" * 50)
        print("NEW HAND")
        print("=" * 50)

    # Play until game over
    while not ai_player.is_game_over():
        game_state = ai_player.get_game_state()

        if verbose:
            print_game_state(ai_player.env, human_player)

        current = ai_player.env.current_player

        if current == human_player:
            # Human's turn
            action = get_human_action(game_state['valid_actions'])
        else:
            # AI's turn
            if verbose:
                print("\nOpponent is thinking...")
            action = ai_player.get_action(state, game_state['valid_actions'])
            if verbose:
                action_names = {0: 'Check/Call', 1: 'Bet/Raise', 2: 'Fold'}
                print(f"Opponent chose: {action_names[action]}")

        # Play
        state, reward, done, info = ai_player.play(action)

    # Result
    if verbose:
        clear_screen()
        print("\n" + "=" * 50)
        print("HAND COMPLETE")
        print("=" * 50)

        # Show final hands
        print(f"Pot: {ai_player.env.pot}")

        player_hand = ai_player.env.hands[human_player]
        opponent_hand = ai_player.env.hands[1 - human_player]

        print(f"Your hand: {' '.join([card_to_string(c) for c in player_hand])}")
        print(f"Opponent: {' '.join([card_to_string(c) for c in opponent_hand])}")

        if ai_player.env.board:
            print(f"Board: {' '.join([card_to_string(c) for c in ai_player.env.board])}")

        print("-" * 50)

        if ai_player.env.winner == human_player:
            print(">>> YOU WIN! <<<")
        elif ai_player.env.winner == 1 - human_player:
            print(">>> YOU LOSE <<<")
        else:
            print(">>> TIE <<<")

    return ai_player.env.winner


def play_continue():
    """Ask to play another hand"""
    print("\nPlay another hand? (y/n): ", end="")
    choice = input().strip().lower()
    return choice in ['y', 'yes', '']


def main():
    """Main function"""
    import argparse
    from aiplayer import AIPlayer

    parser = argparse.ArgumentParser(description='Play poker against NFSP bot')
    parser.add_argument('--model', type=str, default='nfsp_agent_final.pt',
                       help='Path to trained model')
    parser.add_argument('--human-first', action='store_true',
                       help='Human plays first')

    args = parser.parse_args()

    # Create AI player
    ai = AIPlayer()

    # Try to load model
    try:
        ai.load_model(args.model)
        ai.set_evaluate_mode()
        print("\nLoaded trained model!")
    except:
        print("\nWarning: Could not load model. Using untrained agent.")
        print("Train first with: python train.py")

    # Play
    human_player = 0 if args.human_first else 1

    total_wins = 0
    total_losses = 0
    total_ties = 0

    while True:
        winner = play_hand(ai, human_player=human_player)

        if winner == human_player:
            total_wins += 1
        elif winner == 1 - human_player:
            total_losses += 1
        else:
            total_ties += 1

        total = total_wins + total_losses + total_ties
        print(f"\nSession record: {total_wins}W / {total_losses}L / {total_ties}T ({total_wins/total:.1%})")

        if not play_continue():
            break

        clear_screen()

    print("\nThanks for playing!")


if __name__ == "__main__":
    main()
