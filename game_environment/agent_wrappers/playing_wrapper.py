from abc import ABC, abstractmethod

from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.environments import TFPyEnvironment, FlattenObservationsWrapper
from tf_agents.networks import categorical_q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

import numpy as np
import tensorflow as tf

from game_environment.config_files.c51_config import C51Config
from game_environment.trick import Trick


def observation_and_action_constraint_splitter(observation):
    return observation['observation'], observation['legal_moves']


class PlayingWrapper(ABC):

    def __init__(self):

        self.name = "Playing"
        self.eval_playing_env = None
        self.train_playing_env = None
        self.eval_py_playing_env = None
        self.train_py_playing_env = None

        self.policy = None
        self.train_phase = None

        self.current_playing_action_step = None
        self.previous_playing_time_step = None
        self.current_playing_time_step = None

        self.network = None

        self.optimizer = None

        self.train_step_counter = None

        self.agent = None

        self.rewards = []

        self.total_rewards = []

        self.total_avg_rewards = []

        self.total_returns = []

        self.total_avg_returns = []

        self.total_train_losses = []

        self.returns = []

    @abstractmethod
    def train_agent(self):
        return

    def set_playing_env(self, train_env, eval_env):
        self.train_py_playing_env = train_env
        self.eval_py_playing_env = eval_env
        self.train_playing_env = TFPyEnvironment(self.train_py_playing_env)
        self.eval_playing_env = TFPyEnvironment(self.eval_py_playing_env)

    def reset_train_step_counter(self):
        # Reset the train step
        self.agent.train_step_counter.assign(0)

    def reset_rewards(self):
        self.rewards = []

    def play_card(self, current_trick: Trick, playable_cards):
        self.set_current_trick(current_trick)
        self.set_playable_cards(playable_cards)
        if self.train_phase:
            if current_trick.game_round_number == 1:
                self.previous_playing_time_step = None
                self.current_playing_time_step = None
                self.current_playing_action_step = None
                self.train_playing_env.reset()
            return self.playing_train_phase()
        else:
            if current_trick.game_round_number == 1:
                self.previous_playing_time_step = None
                self.current_playing_time_step = None
                self.current_playing_action_step = None
                self.eval_playing_env.reset()
            return self.playing_eval_phase()

    def set_random_policy(self):
        self.policy = random_tf_policy.RandomTFPolicy(self.train_playing_env.time_step_spec(),
                                                      self.train_playing_env.action_spec())

    def set_agent_policy(self):
        self.policy = self.agent.policy

    def set_policy(self, policy):
        self.policy = policy
        self.agent.policy = policy

    def set_current_trick(self, current_trick: Trick):
        if self.train_phase:
            self.train_playing_env.pyenv.envs[0].set_trick(current_trick)
        else:
            self.eval_playing_env.pyenv.envs[0].set_trick(current_trick)

    def set_playable_cards(self, playable_cards):
        if self.train_phase:
            self.train_playing_env.pyenv.envs[0].set_playable_cards(playable_cards)
        else:
            self.eval_playing_env.pyenv.envs[0].set_playable_cards(playable_cards)

    @abstractmethod
    def playing_train_phase(self):
        return

    def playing_eval_phase(self):
        current_time_step = self.eval_playing_env.current_time_step()
        if self.current_playing_action_step is not None:
            self.eval_playing_env.step(self.current_playing_action_step.action)
            current_time_step = self.eval_playing_env.current_time_step()
            self.rewards.append(current_time_step.reward)
        self.current_playing_action_step = self.policy.action(current_time_step)
        return self.current_playing_action_step.action.numpy()[0]
