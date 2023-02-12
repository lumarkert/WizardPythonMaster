from game_environment.players.player import Player
from game_environment.round import Round
from random import randrange

from game_environment.trick import Trick


class RLPlayerBidding(Player):
    def __init__(self, name, directory):
        self.bidding_wrapper = None
        super().__init__(name, directory)

    def set_bidding_wrapper(self, wrapper):
        self.bidding_wrapper = wrapper

    def play_card(self, current_trick: Trick):
        playable_cards = self.calculate_playable_cards(current_trick.trick_suit)
        if len(playable_cards) == 1:
            card_idx = 0
        else:
            card_idx = randrange(0, len(playable_cards))
        hand_idx = playable_cards[card_idx]
        played_card = self.hand[hand_idx]
        if played_card.color != 'NOC':
            self.hand_counted_by_color[played_card.color] -= 1
        print(f'{self.name} played this card: {self.hand[int(card_idx)]}')
        return self.hand.pop(hand_idx)

    def call_tricks(self, game_round):
        self.guessed_tricks = self.bidding_wrapper.bidding(game_round)
        return self.guessed_tricks

    def set_trump_suit(self, deck):
        trump_input = randrange(0, len(deck.colors))
        return deck.colors[int(trump_input)]

    def wrap_up_game(self):
        self.bidding_wrapper.wrap_up_game()

    def set_round_ended(self):
        return

    def reset_game(self):
        self.wrap_up_game()
        self.reset_game_stats()
        return

