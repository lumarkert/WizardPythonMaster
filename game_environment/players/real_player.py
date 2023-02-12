from game_environment.players.player import Player
from game_environment.round import Round
from game_environment.trick import Trick


class RealPlayer(Player):
    def __init__(self, name, directory):
        super().__init__(name, directory)
        self.is_training_agent = False
        
    def play_card(self, current_trick: Trick):
        print('-----------------------------------')
        print(f'Current Player: {self.name}\n')
        print(f'The current Trump Suit is {current_trick.trump_suit.color}\n')
        if len(current_trick.cards_in_trick) == 0:
            print("You are playing the first card of the trick.\n")
        else:
            print("Current Cards in the trick:")
            for player in current_trick.cards_in_trick:
                print(f'{player.name} played {current_trick.cards_in_trick[player]}')
            print('')

        playable_cards = self.calculate_playable_cards(current_trick.trick_suit)
        self.print_hand()
        self.print_playable_cards(playable_cards)

        card_idx = input(f'\n{self.name}, it is your turn. Which Card do you want to play? Enter the corresponding number.')

        while not card_idx.isdigit() or int(card_idx) < 0 or (not (int(card_idx)-1) in playable_cards):
            card_idx = input('Please enter a valid number.')
        played_card = self.hand[int(card_idx) - 1]
        print(f'{self.name} played this card: {played_card}')
        if played_card.color != 'NOC':
            self.hand_counted_by_color[played_card.color] -= 1
        return self.hand.pop(int(card_idx) - 1)

    def call_tricks(self, game_round):
        print('-----------------------------------')
        print(f'Current Player: {self.name}\n')
        print('')
        print(f'The current Trump Suit is {game_round.trump_suit.color}\n')
        if len(game_round.called_tricks) == 0:
            print(f'You are the first player to call tricks!\n')

        else:
            print('These are the Tricks of your opponents')
            for called_trick in game_round.called_tricks:
                print(f'{called_trick} called {game_round.called_tricks[called_trick]} tricks!')
            print('')
        self.print_hand()
        trick_input = input(f'\nHow many tricks are u calling?')
        while not trick_input.isdigit() or int(trick_input) < 0 or int(trick_input) > len(self.hand):
            trick_input = input('Please enter a valid number.')
        self.guessed_tricks = trick_input
        return trick_input

    def set_trump_suit(self, deck):

        print(f'\n{self.name}, a wizard was drawn as trump suit. You can decide which color will be the trump suit.\n')
        self.print_hand()
        colors_string=""
        print('')
        for idx, color in enumerate(deck.colors):
            colors_string += f'[{idx+1}: {color}] '
        print(colors_string)
        trump_input = input('\nWhich color should be the trump suit this round. Please enter the corresponding number.')
        while not trump_input.isdigit() or int(trump_input) < 0 or int(trump_input) > len(deck.colors):
            trump_input = input('Please enter a valid number.')
        return deck.colors[int(trump_input) - 1]

    def wrap_up_game(self):
        return

    def wrap_up_round(self, current_trick):
        return

    def set_round_ended(self):
        return

    def reset_game(self):
        self.won_tricks.append(self.won_tricks)
        self.wrap_up_game()
        self.reset_game_stats()
        return
