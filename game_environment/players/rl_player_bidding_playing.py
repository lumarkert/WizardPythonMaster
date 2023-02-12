from random import randrange

from game_environment.agent_wrappers.bidding_wrapper import BiddingWrapper
from game_environment.agent_wrappers.playing_wrapper import PlayingWrapper
from game_environment.players.player import Player
from game_environment.agent_wrappers.c51_bidding_wrapper import C51BiddingWrapper
from game_environment.agent_wrappers.c51_playing_wrapper import C51PlayingWrapper

from game_environment.trick import Trick


class RLPlayerBiddingPlaying(Player):
    def __init__(self, name, directory):
        self.bidding_wrapper: BiddingWrapper = None
        self.playing_wrapper: PlayingWrapper = None
        super().__init__(name, directory)

    def set_bidding_wrapper(self, wrapper):
        self.bidding_wrapper = wrapper

    def set_playing_wrapper(self, wrapper):
        self.playing_wrapper = wrapper

    def play_card(self, current_trick: Trick):
        playable_cards = self.calculate_playable_cards(current_trick.trick_suit)
        card_id = self.playing_wrapper.play_card(current_trick, playable_cards)

        hand_idx = self.get_hand_idx_from_card_id(card_id)
        played_card = self.hand[hand_idx]
        if played_card.color != 'NOC':
            self.hand_counted_by_color[played_card.color] -= 1
        print(f'{self.name} played this card: {self.hand[int(hand_idx)]}')
        return self.hand.pop(hand_idx)

    def get_hand_idx_from_card_id(self, card_id):
        for idx, card in enumerate(self.hand):
            if card.card_id == card_id:
                return idx

    def call_tricks(self, game_round):
        self.guessed_tricks = self.bidding_wrapper.bidding(game_round)
        return self.guessed_tricks

    def set_trump_suit(self, deck):
        trump_input = randrange(0, len(deck.colors))
        return deck.colors[int(trump_input)]

    def set_round_ended(self):
        self.playing_wrapper.set_round_ended()

    def wrap_up_game(self, current_trick):
        self.bidding_wrapper.wrap_up_game()
        playable_cards = self.calculate_playable_cards(current_trick.trick_suit)
        self.playing_wrapper.wrap_up_game(playable_cards, current_trick)

    def reset_game(self):
        self.wrap_up_game()
        self.reset_game_stats()
