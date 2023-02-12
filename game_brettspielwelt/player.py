from abc import ABC, abstractmethod

import numpy as np

from game_brettspielwelt.round import Round
from game_brettspielwelt.trick import Trick


class Player(ABC):
    def __init__(self, name):
        self.name = name
        self.hand = []
        self.played_cards = []
        self.guessed_tricks = -1
        self.won_tricks = 0
        self.points = 0
        self.hand_counted_by_color = {}
        self.points_per_round = []
        self.is_training_agent = None

        self.accuracy = []
        self.avg_accuracies = []
        self.avg_points_per_round = []
        self.avg_accuracy_per_round = []

        self.history_points_per_round = []
        self.history_accuracy_per_round = []

        self.history_points = []
        self.avg_points = []

        self.games_won = 0
        self.winrates = []



    @abstractmethod
    def play_card(self, current_trick: Trick):
        return

    @abstractmethod
    def call_tricks(self, game_round):
        return

    @abstractmethod
    def set_trump_suit(self, deck):
        return

    @abstractmethod
    def reset_game(self):
        return

    @abstractmethod
    def wrap_up_game(self):
        return

    @abstractmethod
    def wrap_up_round(self, current_trick):
        return

    def calculate_playable_cards(self, trick_suit):
        if trick_suit is None or self.hand_counted_by_color.get(trick_suit) == 0:
            return list(range(0, len(self.hand)))
        else:
            playable_cards = []
            for idx, card in enumerate(self.hand):
                if card.value > 13 or card.color == trick_suit:
                    playable_cards.append(idx)
            return playable_cards

    def count_hand_by_color(self, deck):
        for color in deck.colors:
            self.hand_counted_by_color[color] = 0
        for card in self.hand:
            if card.value <= 13:
                self.hand_counted_by_color[card.color] = self.hand_counted_by_color.get(card.color) + 1

    def print_hand(self):
        print('This is your hand:')
        hand_string = ""
        for idx, card in enumerate(self.hand):
            hand_string += f'[{idx+1}: {card}] '
        print(hand_string)

    def print_playable_cards(self, playable_cards):
        print('\nBut you can only play the following cards:')
        hand_string = ""
        for card_idx in playable_cards:
            hand_string += f'[{int(card_idx)+1}: {self.hand[card_idx]}] '
        print(hand_string)

    def add_card_to_hand(self, card):
        self.hand.append(card)

    def increment_won_tricks(self):
        self.won_tricks += 1

    def reset_tricks(self):
        self.guessed_tricks = -1
        self.won_tricks = 0

    def reset_played_cards(self):
        self.played_cards = []

    def reset_stats(self):
        self.guessed_tricks = -1
        self.won_tricks = 0
        self.points = 0
        self.points_per_round = []

    def reset_win_stats(self):
        self.games_won = 0
        self.accuracy = []
        self.history_points = []
        self.history_points_per_round = []

    def score_points(self):
        point_change = 0
        self.calculate_accuracy()
        if int(self.guessed_tricks) == int(self.won_tricks):
            point_change += (20 + int(self.guessed_tricks) * 10)
        else:
            point_change -= abs(int(self.guessed_tricks) - int(self.won_tricks)) * 10
        self.reset_tricks()
        print(f'Score Change this Round: {point_change}')
        self.points_per_round.append(point_change)
        self.points += point_change
        print(f'New Score: {self.points}')

    def calculate_accuracy(self):
        if int(self.guessed_tricks) == int(self.won_tricks):
            self.accuracy.append(100)
        else:
            self.accuracy.append(0)

    def add_card_to_played_cards(self, card):
        self.played_cards.append(card)

    def calculate_win_rate(self, total_games):
        return (self.games_won / total_games) * 100

    def calculate_stats_per_round(self, total_rounds):
        points_sorted_after_round_number = []
        accuracy_sorted_after_round_number = []
        for idx in range(total_rounds):
            points_sorted_after_round_number.append([])
            accuracy_sorted_after_round_number.append([])
        for idx in range(len(self.history_points_per_round)):
            round_number = idx % total_rounds
            points_sorted_after_round_number[round_number].append(self.history_points_per_round[idx])
            accuracy_sorted_after_round_number[round_number].append(self.accuracy[idx])

        for idx in range(total_rounds):
            if len(self.avg_points_per_round) < total_rounds:
                self.avg_points_per_round.append([])
            if len(self.avg_accuracy_per_round) < total_rounds:
                self.avg_accuracy_per_round.append([])
            self.avg_points_per_round[idx].append((sum(points_sorted_after_round_number[idx]) / len(points_sorted_after_round_number[idx])))
            self.avg_accuracy_per_round[idx].append((sum(accuracy_sorted_after_round_number[idx]) / len(accuracy_sorted_after_round_number[idx])))



    def calculate_average_accuracies(self):
        self.avg_accuracies.append(sum(self.accuracy) / len(self.accuracy))




