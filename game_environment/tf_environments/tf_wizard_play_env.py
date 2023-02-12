from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts
from tf_agents.environments.utils import validate_py_environment
import numpy as np

import game_state_converter as gsc
from game_environment.config_files.c51_config import C51Config
from game_environment.config_files.game_config import GameConfig


# https://github.com/igilitschenski/multi_car_racing/blob/master/gym_multi_car_racing/multi_car_racing.py


class TFWizardPlayEnv(py_environment.PyEnvironment):

    def __init__(self, game_config: GameConfig, c51_config: C51Config, player):
        self.playable_cards = None
        self.current_trick = None
        self.player = player
        self.input_mode = c51_config.environment
        self.number_of_cards = (game_config.number_of_values * len(
            game_config.colors)) + game_config.number_of_wizards + game_config.number_of_jesters
        self.number_of_colors = game_config.number_of_colors
        self.number_of_rounds = game_config.number_of_total_rounds
        if self.input_mode == "full_ohe":
            self._observation_spec = self.determine_observation_spec_full_ohe(game_config)
        elif self.input_mode == "small_input":
            self._observation_spec = self.determine_observation_spec_small_input(game_config.number_of_colors)
        else:
            self._observation_spec = self.determine_observation_spec(game_config.number_of_colors)

        self._action_spec = self.determine_action_spec()
        self._state = np.zeros((2 * self.number_of_cards + 7))
        self._episode_ended = False
        self.game_ended = False
        self.round_ended = False
        self.invalid_action = False
        self.reward_function = c51_config.reward_function

    def determine_action_spec(self):
        return array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=self.number_of_cards - 1,
                                           name='action')

    def determine_observation_spec_full_ohe(self, game_config: GameConfig):

        number_of_inputs = 3 * self.number_of_cards + (2 * game_config.number_of_total_rounds) + (2 * (game_config.number_of_total_rounds + 1)) + (
                    game_config.number_of_colors + 1) + game_config.num_players + (
            2 * ((game_config.num_players - 1) * (game_config.number_of_total_rounds + 1)))

        # 'own_hand': self.number_of_cards 60
        # 'played_cards': self.number_of_cards 60
        # 'trump_color': self.colors.length + 1 5
        # 'cards_in_current_trick': self.number_of_cards 60
        # 'round_number': number_of_rounds, 15
        # 'trick_number': number_of_rounds, 15
        # 'current_position_in_round': number_of_players, 4
        # 'own_called_tricks': number_of_rounds + 1, 16
        # 'own_current_tricks': number_of_rounds+  1, 16
        # 'opponents_called_tricks': players-1 * number_of_rounds,  48
        # 'opponents_current_tricks': players-1 * number_of_rounds, 48

        return {'observation': array_spec.BoundedArraySpec(shape=(number_of_inputs,), dtype=np.int32),
                'legal_moves': array_spec.BoundedArraySpec(shape=(self.number_of_cards,), dtype=np.int32)}

    def determine_observation_spec_small_input(self, number_of_colors):

        number_of_inputs = 3 * self.number_of_cards + (number_of_colors + 1) + 5

        # 'own_hand': array_spec.BoundedArraySpec((1, self.number_of_cards,), np.int32, minimum=0,
        #                                         maximum=1),
        # 'played_cards': array_spec.BoundedArraySpec((1, self.number_of_cards,), np.int32, minimum=0,
        #                                                 maximum=1),
        # 'trump_color': array_spec.BoundedArraySpec((1, self.colors.length), np.int32, minimum=0,
        #                                           maximum=1),
        # 'cards_in_current_trick': array_spec.BoundedArraySpec((1, self.number_of_cards,), np.int32, minimum=0,
        #                                                      maximum=1),
        # 'round_number': array_spec.BoundedArraySpec((1,), np.int32, minimum=-1,
        #                                             maximum=number_of_total_rounds),
        # 'trick_number': array_spec.BoundedArraySpec((1,), np.int32, minimum=-1,
        #                                             maximum=number_of_total_rounds),
        # 'current_position_in_round': array_spec.BoundedArraySpec((1,), np.int32, minimum=-1,
        #                                                          maximum=(num_players - 1)),
        # 'own_called_tricks': array_spec.BoundedArraySpec((1,), np.int32, minimum=-1,
        #                                                  maximum=number_of_total_rounds),
        # 'own_current_tricks': array_spec.BoundedArraySpec((1,), np.int32, minimum=-1,
        #                                                   maximum=number_of_total_rounds),
        return {'observation': array_spec.BoundedArraySpec(shape=(number_of_inputs,), dtype=np.int64),
                'legal_moves': array_spec.BoundedArraySpec(shape=(self.number_of_cards,), dtype=np.int32)}

    def determine_observation_spec(self, game_config):
        number_of_inputs = (5 + game_config.num_players - 1) * self.number_of_cards + 6 + (
                    3 * (game_config.num_players - 1))
        # 'own_hand': number_of_cards
        # 'own_played_cards': number_of_cards
        # 'opponents_played_cards': players * number_of_cards
        # 'trump_card': number_of_cards
        # 'highest_card_in_trick': number_of_cards
        # 'cards_in_current_trick': number_of_cards
        # 'round_number': 1
        # 'current_trick': 1
        # 'current_position_in_round': 1
        # 'own_called_tricks': 1
        # 'own_current_tricks': 1
        # 'own_achieved_points': 1
        # 'opponents_called_tricks': players - 1
        # 'opponents_current_tricks': players - 1
        # 'opponents_achieved_points': players - 1
        return {'observation': array_spec.BoundedArraySpec(shape=(number_of_inputs,), dtype=np.int64),
                'legal_moves': array_spec.BoundedArraySpec(shape=(self.number_of_cards,), dtype=np.int32)}

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = self.choose_state_calculation()

        self._episode_ended = False
        return ts.restart(self._state)

    def _step(self, action):
        self._state = self.choose_state_calculation()

        if self.round_ended:
            reward = self.choose_reward_function(self.player.points_per_round[-1])
            self.round_ended = False
        else:
            reward = 0

        if self._episode_ended:
            return ts.termination(self._state, reward)
        else:
            return self.create_transition(self._state, reward, 1.0)

    def choose_state_calculation(self):
        if self.input_mode == "full_ohe":
            return gsc.calculate_playing_state_full_ohe(self.number_of_cards, self.player,
                                                        self.current_trick,
                                                        self.playable_cards, self.number_of_colors,
                                                        self.number_of_rounds)
        elif self.input_mode == "small_input":
            return gsc.calculate_playing_state_small_input(self.number_of_cards, self.player,
                                                           self.current_trick,
                                                           self.playable_cards, self.number_of_colors)
        else:
            return gsc.calculate_playing_state(self.number_of_cards, self.player, self.current_trick, self.playable_cards)

    def choose_reward_function(self, points_last_round):
        if self.reward_function == "binary_reward":
            return self.binary_reward(points_last_round)
        if self.reward_function == "no_negative_reward":
            return self.no_negative_reward(points_last_round)
        if self.reward_function == "points_as_reward":
            return self.points_as_reward(points_last_round)
        if self.reward_function == "points_as_reward_times_10":
            return self.points_as_reward_times_10(points_last_round)
        if self.reward_function == "normalize_reward":
            return self.normalize_reward(points_last_round, self.current_trick.game_round_number)

    @staticmethod
    def binary_reward(points_last_round):
        if points_last_round > 0:
            return 1
        else:
            return 0

    @staticmethod
    def normalize_reward(points_last_round, current_round_number):
        if points_last_round > 0:
            max_points = (current_round_number * 10) + 20
            return points_last_round / max_points
        else:
            minimum_points = (current_round_number * 10)
            return points_last_round / minimum_points

    @staticmethod
    def no_negative_reward(points_last_round):
        if points_last_round > 0:
            return points_last_round
        else:
            return 0

    @staticmethod
    def points_as_reward(points_last_round):
        return points_last_round

    @staticmethod
    def points_as_reward_times_10(points_last_round):
        return points_last_round * 10

    @staticmethod
    def create_transition(state, reward, discount):
        return ts.transition(state, reward, discount)

    def set_trick(self, trick):
        self.current_trick = trick

    def set_playable_cards(self, playable_cards):
        self.playable_cards = playable_cards

    def set_episode_ended(self, episode_ended):
        self._episode_ended = episode_ended

    def set_round_ended_true(self):
        self.round_ended = True
