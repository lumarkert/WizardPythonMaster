import logging
from abc import ABC, abstractmethod

from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.environments import TFPyEnvironment, FlattenObservationsWrapper
from tf_agents.networks import categorical_q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

import tensorflow as tf

from game_environment.config_files.c51_config import C51Config
from game_environment.round import Round


class BiddingWrapper(ABC):

    def __init__(self):

        self.name = "Bidding"
        self.eval_bidding_env = None
        self.train_bidding_env = None
        self.eval_py_bidding_env = None
        self.train_py_bidding_env = None

        self.policy = None
        self.train_phase = None

        self.current_bidding_action_step = None
        self.previous_bidding_time_step = None
        self.current_bidding_time_step = None

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

        self.returns =[]

    @abstractmethod
    def train_agent(self):
        return

    def set_bidding_env(self, train_env, eval_env):
        self.train_py_bidding_env = train_env
        self.eval_py_bidding_env = eval_env
        self.train_bidding_env = TFPyEnvironment(self.train_py_bidding_env)
        self.eval_bidding_env = TFPyEnvironment(self.eval_py_bidding_env)

    def reset_train_step_counter(self):
        self.agent.train_step_counter.assign(0)

    def reset_rewards(self):
        self.rewards = []

    def set_random_policy(self):
        self.policy = random_tf_policy.RandomTFPolicy(self.train_bidding_env.time_step_spec(),
                                                      self.train_bidding_env.action_spec())

    def set_agent_policy(self):
        self.policy = self.agent.policy

    def set_policy(self, policy):
        self.policy = policy
        self.agent.policy = policy

    def set_game_round(self, game_round: Round):
        if self.train_phase:
            self.train_bidding_env.pyenv.envs[0].set_round(game_round)
        else:
            self.eval_bidding_env.pyenv.envs[0].set_round(game_round)

    def bidding(self, game_round: Round):
        self.set_game_round(game_round)
        if self.train_phase:
            if game_round.round_number == 1:
                self.previous_bidding_time_step = None
                self.current_bidding_time_step = None
                self.current_bidding_action_step = None
                self.train_bidding_env.reset()
            return self.bidding_train_phase()
        else:
            if game_round.round_number == 1:
                self.previous_bidding_time_step = None
                self.current_bidding_time_step = None
                self.current_bidding_action_step = None
                self.eval_bidding_env.reset()
            return self.bidding_eval_phase()

    @abstractmethod
    def bidding_train_phase(self):
        return

    def bidding_eval_phase(self):
        current_time_step = self.eval_bidding_env.current_time_step()
        if self.current_bidding_action_step is not None:
            self.eval_bidding_env.step(self.current_bidding_action_step.action)
            current_time_step = self.eval_bidding_env.current_time_step()
            self.rewards.append(current_time_step.reward)
        self.current_bidding_action_step = self.policy.action(current_time_step)
        return self.current_bidding_action_step.action.numpy()[0]



    @abstractmethod
    def wrap_up_game(self):
        return









