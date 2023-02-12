from game_brettspielwelt.player import Player
from game_brettspielwelt.round import Round
from game_brettspielwelt.trick import Trick


class BrettspielweltPlayer(Player):
    def __init__(self, name):
        super().__init__(name)
        self.is_training_agent = False
        
    def play_card(self, current_trick: Trick):
        card_string = input(f'\nWhich Card did the player {self.name} play?')
        card_idx = self.determine_card(card_string, current_trick)
        while card_idx is None:
            card_string = input('Please enter a valid combination.')
            card_idx = self.determine_card(card_string, current_trick)
        current_trick.deck.cards[int(card_idx)]
        played_card = current_trick.deck.cards[int(card_idx)]
        print(f'{self.name} played this card: {played_card}')
        return played_card

    def determine_card(self, card_string, current_trick):
        if len(card_string) < 2:
            return None
        color = card_string[0]
        if not color.isalpha() or (color != "y" and color != "r" and color != "b" and color != "g" and color != "w" and color != "j"):
            return None
        if len(card_string) == 3:
            value = card_string[1:3]
        else:
            value = card_string[1]
        if (color == "w" or color == "j") and int(value) > current_trick.deck.number_of_wizards:
            return None
        if not value.isdigit() or int(value) > current_trick.deck.number_of_values:
            return None
        value = int(value)
        card_start_idx = 0
        if color == "g":
            card_start_idx = 0
        elif color == "b":
            card_start_idx = 13
        elif color == "y":
            card_start_idx = 26
        elif color == "r":
            card_start_idx = 39
        elif color == "w":
            card_start_idx = 52
        elif color == "j":
            card_start_idx = 56

        return card_start_idx + value - 1

    def call_tricks(self, game_round):
        trick_input = input(f'\nHow many tricks is player {self.name} calling?')
        while not trick_input.isdigit() or int(trick_input) < 0 or int(trick_input) > game_round.round_number:
            trick_input = input('Please enter a valid number.')
        self.guessed_tricks = int(trick_input)
        return int(trick_input)

    def set_trump_suit(self, deck):
        print(f'\nA wizard was drawn as trump suit.\n')
        colors_string = ""
        print('')
        for idx, color in enumerate(deck.colors):
            colors_string += f'[{idx + 1}: {color}] '
        print("Available Colors:")
        print(colors_string)
        trump_input = input(f'\nWhich color has player {self.name} picked?')
        while not trump_input.isdigit() or int(trump_input) < 0 or int(trump_input) > len(deck.colors):
            trump_input = input('Please enter a valid number.')
        return deck.colors[int(trump_input) - 1]

    def wrap_up_game(self):
        return

    def wrap_up_round(self, current_trick):
        return

    def reset_game(self):
        self.wrap_up_game()
        self.reset_stats()
        return
