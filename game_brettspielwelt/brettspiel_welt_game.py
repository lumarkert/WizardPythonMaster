import copy
from random import randrange

import numpy as np

from game_environment.config_files.game_config import GameConfig
from game_brettspielwelt.deck import Deck
from game_brettspielwelt.round import Round


class BrettspielweltGame:

    def __init__(self, players, game_config: GameConfig):
        self.players = players
        self.number_of_players = len(self.players)
        self.current_round = 0
        self.deck = Deck(game_config.colors, game_config.number_of_values, game_config.number_of_wizards,
                         game_config.number_of_jesters)
        self.start_player_idx = int(input("Which Player is starting?"))
        self.game_round = None
        if game_config.number_of_total_rounds == 0:
            self.number_of_rounds = int(self.deck.number_of_cards / self.number_of_players)
        else:
            self.number_of_rounds = game_config.number_of_total_rounds
        self.game_ended = False

    def reset_game_for_players(self):
        for player in self.players:
            player.reset_stats()

    def start_game(self):
        self.reset_game_for_players()
        for _ in range(self.number_of_rounds):
            self.create_round()
            self.play_round()
            self.start_player_idx = (self.start_player_idx + 1) % len(self.players)
        self.wrap_up_game()

    def wrap_up_game(self):
        for player in self.players:
            player.wrap_up_game()
            player.history_points.append(player.points)
            player.history_points_per_round = np.concatenate((player.history_points_per_round, player.points_per_round),
                                                             axis=None)
        self.determine_winner()

    def create_round(self):
        self.current_round += 1
        print('')
        print(f'Starting Round number {self.current_round}')
        print('')
        self.game_round = Round(self.players, self.start_player_idx, self.current_round, self.deck)

    def play_round(self):
        self.game_round.play_round()

    def determine_winner(self):
        winning_player = None
        for player in self.players:
            if (winning_player is None) or player.points > winning_player.points:
                winning_player = player
        winning_player.games_won += 1
