from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

import game_state_converter as gsc
from game_environment.config_files.c51_config import C51Config
from game_environment.config_files.game_config import GameConfig


# https://github.com/igilitschenski/multi_car_racing/blob/master/gym_multi_car_racing/multi_car_racing.py


class TFWizardBiddingEnv(py_environment.PyEnvironment):

    def __init__(self, game_config: GameConfig, c51_config: C51Config, player):
        self.current_round = None
        self.player = player
        self.input_mode = c51_config.environment
        self.number_of_cards = (game_config.number_of_values * len(
            game_config.colors)) + game_config.number_of_wizards + game_config.number_of_jesters
        self.total_number_of_rounds = game_config.number_of_total_rounds

        if self.input_mode == "full_ohe":
            self._observation_spec = self.determine_observation_spec_full_ohe(game_config)
        else:
            self._observation_spec = self.determine_observation_spec(game_config)

        self._action_spec = self.determine_action_spec()
        self._state = np.zeros((2 * self.number_of_cards + 7))
        self._episode_ended = False
        self.reward_function = c51_config.reward_function

    def determine_action_spec(self):
        return array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=self.total_number_of_rounds, name='action')

    def determine_observation_spec_full_ohe(self, game_config):
        number_of_inputs = (2 * self.number_of_cards) + self.total_number_of_rounds + game_config.num_players + (game_config.num_players - 1) * (self.total_number_of_rounds + 1)
        #     'own_hand': self.number_of_cards 60
        #     'trump_card': self.number_of_cards 60
        #     'round_number': number_of_rounds 15
        #     'current_position_in_round': number_of_players 4
        #     'opponents_called_tricks': number_of_players - 1 * (number_of_rounds + 1) 45
        return {'observation': array_spec.BoundedArraySpec(shape=(number_of_inputs,), dtype=np.int32),
                'legal_moves': array_spec.BoundedArraySpec(shape=(self.total_number_of_rounds + 1,), dtype=np.int32)}
    def determine_observation_spec(self, game_config):
        number_of_inputs = 2 * self.number_of_cards + 3 + 2 * (game_config.num_players - 1)
        #     'own_hand': self.number_of_cards
        #     'trump_card': self.number_of_cards
        #     'round_number': 1
        #     'current_position_in_round': 1
        #     'opponents_called_tricks': players - 1
        #     'own_achieved_points': 1
        #     'opponents_achieved_points': players - 1
        return {'observation': array_spec.BoundedArraySpec(shape=(number_of_inputs,), dtype=np.int64),
                'legal_moves': array_spec.BoundedArraySpec(shape=(self.total_number_of_rounds + 1,), dtype=np.int32)}
        # return {
        #     'own_hand': array_spec.BoundedArraySpec((self.number_of_cards,), np.int32, minimum=0,
        #                                             maximum=1),
        #     'trump_card': array_spec.BoundedArraySpec((self.number_of_cards,), np.int32, minimum=0,
        #                                               maximum=1),
        #     'round_number': array_spec.BoundedArraySpec((), np.int32, minimum=-1,
        #                                                 maximum=number_of_total_rounds),
        #     'current_position_in_round': array_spec.BoundedArraySpec((), np.int32, minimum=-1,
        #                                                              maximum=(num_players - 1)),
        #     'opponent1_called_tricks': array_spec.BoundedArraySpec((), np.int32, minimum=-1,
        #                                                            maximum=number_of_total_rounds),
        #     'opponent2_called_tricks': array_spec.BoundedArraySpec((), np.int32, minimum=-1,
        #                                                            maximum=number_of_total_rounds),
        #     'own_achieved_points': array_spec.BoundedArraySpec((), np.int32, minimum=-2000,
        #                                                        maximum=2000),
        #     'opponent1_achieved_points': array_spec.BoundedArraySpec((), np.int32, minimum=-2000,
        #                                                              maximum=2000),
        #     'opponent2_achieved_points': array_spec.BoundedArraySpec((), np.int32, minimum=-2000,
        #                                                              maximum=2000)
        # }

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
        reward = self.choose_reward_function(self.player.points_per_round[-1])
        if self._episode_ended:
            reward = reward + self.player.points
            return ts.termination(self._state, reward)
        else:
            return ts.transition(self._state, reward=reward, discount=1.0)

    def choose_state_calculation(self):
        if self.input_mode == "full_ohe":
            return gsc.calculate_bidding_state_full_ohe(self.number_of_cards, self.player, self.current_round, self.total_number_of_rounds)
        else:
            return gsc.calculate_bidding_state(self.number_of_cards, self.player, self.current_round, self.total_number_of_rounds)

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
            return self.normalize_reward(points_last_round, self.current_round.round_number)

    @staticmethod
    def normalize_reward(points_last_round, current_round_number):
        if points_last_round > 0:
            max_points = (current_round_number * 10) + 20
            return points_last_round / max_points
        else:
            minimum_points = (current_round_number * 10)
            return points_last_round / minimum_points

    @staticmethod
    def binary_reward(points_last_round):
        if points_last_round > 0:
            return 1
        else:
            return 0

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

    def set_round(self, game_round):
        self.current_round = game_round

    def set_episode_ended(self, episode_ended):
        self._episode_ended = episode_ended
