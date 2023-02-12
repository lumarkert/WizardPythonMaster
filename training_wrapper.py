from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time as t

import numpy as np
import tensorflow as tf

import save_manager
from game_brettspielwelt.brettspiel_welt_game import BrettspielweltGame
from game_brettspielwelt.brettspielwelt_player import BrettspielweltPlayer
from game_environment.game import Game
from player_manager import PlayerManager
from training_setup import TrainingSetup


class TrainingWrapper:
    def __init__(self, training_setup: TrainingSetup):
        print(tf.version.VERSION)

        self.start_time = t.time()
        self.start_of_cycle = None
        self.directory = save_manager.determine_save_file_path(training_setup)

        self.training_config = training_setup.training_config
        self.training_setup = training_setup
        self.game_config = training_setup.game_config

        self.player_manager = PlayerManager(training_setup, self.directory)
        self.training_wrappers = self.player_manager.training_wrappers
        self.wrapper_list = self.player_manager.wrapper_list
        self.players = self.player_manager.players
        self.player = self.player_manager.player
        self.start_step = self.player.bidding_wrapper.agent.train_step_counter.numpy()

    def start_training(self):
        #self.bsw_test(self.wrapper_list)
        #self.bsw_test(self.wrapper_list)
        big_plot_steps = [int(self.training_config.num_iterations / 2 + self.start_step),
                          int(self.start_step + self.training_config.num_iterations)]
        self.collect_initial_steps(self.wrapper_list, self.training_config.initial_collect_games)

        for wrapper in self.training_wrappers:
            wrapper.print_current_replay_buffer_size()

        step = self.start_step
        # Evaluate the agent's policy once before training.
        self.collect_steps_for_evaluation(self.wrapper_list, self.training_config.num_eval_episodes)
        self.collect_data_from_evaluation_phase(self.wrapper_list, step)
        self.calculate_stats_per_round_for_players()

        lap_start_time = t.time()

        for _ in range(self.training_config.num_iterations):

            # Collect a few steps using collect_policy and save to the replay buffer.
            if step % self.training_config.collect_games_interval == 0:
                self.collect_steps_for_training(self.wrapper_list, self.training_config.collect_steps_per_iteration)

            # Sample a batch of data from the buffer and update the agent's network.

            for wrapper in self.training_wrappers:
                wrapper.print_current_replay_buffer_size()
                wrapper.train_agent()

            step = self.player.bidding_wrapper.agent.train_step_counter.numpy()

            if step % self.training_config.log_interval == 0:
                for wrapper in self.training_wrappers:
                    print(f'{wrapper.name}:  step = {step}: loss = {wrapper.total_train_losses[-1]}')

            if step % self.training_config.eval_interval == 0:
                self.collect_steps_for_evaluation(self.wrapper_list, self.training_config.num_eval_episodes)
                self.collect_data_from_evaluation_phase(self.wrapper_list, step)
                # self.set_policies_to_best_policy(self.wrapper_list)
                current_time = t.time()
                time_for_last_phase = current_time - lap_start_time
                print(f'Time took for last Train/Evaluation Phase:')
                print(f'{time_for_last_phase} seconds')
                remaining_duration = time_for_last_phase * (
                        (
                                    self.start_step + self.training_config.num_iterations - step) / self.training_config.eval_interval)
                print(f'Remaining Duration')
                print(f'{save_manager.convert_seconds(remaining_duration)}')
                print("-------------------------")
                lap_start_time = current_time

            if big_plot_steps.count(step) > 0:
                self.calculate_stats_per_round_for_players()

            if step % self.training_config.decay_interval == 0:
                print(f'Decaying Variables')
                for wrapper in self.training_wrappers:
                    wrapper.adjust_learning_rate(step)
                    wrapper.adjust_epsilon_greedy(step)
                print("-------------------------")

        save_manager.save_files(self.directory, self.start_step, self.player_manager.wrapper_list,
                                self.player_manager.training_wrappers,
                                self.player_manager.training_setup, self.player_manager.bid_config,
                                self.player_manager.play_config, self.player_manager.game_config,
                                self.player_manager.training_config, self.player_manager.players, self.start_time)

    def bsw_test(self, wrapper_list):
        print("-------------------------")
        print(f"Test Spiel BSW")
        for wrapper in wrapper_list:
            wrapper.set_train_phase(False)
            wrapper.set_agent_policy()
        bsw_players = [self.player]
        for i in range(self.game_config.num_players - 1):
            bsw_players.append(BrettspielweltPlayer(f"BSW_Player_{i + 2}"))
        self.start_bsw_game(bsw_players)

    def collect_data_from_evaluation_phase(self, wrapper_list, step):
        self.calculate_average_points_for_players()
        self.calculate_avg_accuracy_for_players()
        self.calculate_average_reward_and_return_for_wrappers(wrapper_list, step)
        self.calculate_winrate_for_players()

    def calculate_average_points_for_players(self):
        for player in self.players:
            player.avg_points.append(sum(player.history_scores) / len(player.history_scores))

    def calculate_avg_accuracy_for_players(self):
        for player in self.players:
            player.calculate_average_accuracies()

    def calculate_winrate_for_players(self):
        total_games = 0
        for player in self.players:
            total_games += player.games_won

        for player in self.players:
            player.winrates.append(player.calculate_win_rate(total_games))

    def calculate_average_reward_and_return_for_wrappers(self, wrapper_list, step):
        for wrapper in wrapper_list:
            avg_reward_list = []
            for reward_list in wrapper.total_rewards:
                avg_reward_list.append(self.calculate_average_reward(reward_list))
            avg_reward = self.calculate_average_reward(avg_reward_list)
            wrapper.total_avg_rewards.append(avg_reward)
            wrapper.total_rewards = []

            avg_return = sum(wrapper.total_returns)/len(wrapper.total_returns)
            wrapper.total_avg_returns.append(avg_return)
            wrapper.total_returns = []

            print(f'{wrapper.name}:  step = {step}: Average Return = {avg_return}')
            if len(wrapper.total_avg_returns) > 1:
                last_avg_return = wrapper.total_avg_returns[-2]
                print(f'Change from last Evaluation Phase: {avg_return - last_avg_return}')

    def calculate_stats_per_round_for_players(self):
        for player in self.players:
            player.calculate_stats_per_round(self.game_config.number_of_total_rounds)

    def reset_win_stats(self):
        for player in self.players:
            player.reset_evaluation_stats()

    @staticmethod
    def calculate_average_reward(reward_array):
        avg_reward = 0
        for reward in reward_array:
            avg_reward += np.average(reward)
        avg_reward = avg_reward / len(reward_array)
        return avg_reward

    def collect_initial_steps(self, c51_wrapper_list, collect_games):
        print("Collecting Initial Steps")
        sys.stdout = open(os.devnull, 'w')
        for c51_wrapper in c51_wrapper_list:
            c51_wrapper.set_train_phase(True)
            c51_wrapper.set_agent_policy()
        self.reset_win_stats()
        for _ in range(collect_games):
            self.start_game()
        sys.stdout = sys.__stdout__
        print("Finished Initial Steps")

    def collect_steps_for_training(self, c51_wrapper_list, collect_games):
        for c51_wrapper in c51_wrapper_list:
            c51_wrapper.set_train_phase(True)
            c51_wrapper.set_agent_policy()
        sys.stdout = open(os.devnull, 'w')
        self.reset_win_stats()
        for _ in range(collect_games):
            self.start_game()

        sys.stdout = sys.__stdout__


    def collect_steps_for_evaluation(self, wrapper_list, eval_episodes):
        print("-------------------------")
        print(f"Evaluating Setup {self.training_setup.name}")
        print("Collecting Steps for Evaluation")
        for wrapper in wrapper_list:
            wrapper.set_train_phase(False)
            wrapper.set_agent_policy()
        sys.stdout = open(os.devnull, 'w')
        self.reset_win_stats()
        for _ in range(eval_episodes):
            self.start_game()
            for wrapper in wrapper_list:
                wrapper.total_rewards.append(wrapper.rewards)
                wrapper.save_return_of_episode()
                wrapper.reset_rewards()
        sys.stdout = sys.__stdout__
        print("Finished collecting Steps for Evaluation")

    def start_game(self):
        game = Game(self.players,
                    self.game_config)
        game.start_game()

    def start_bsw_game(self, players):
        game = BrettspielweltGame(players,
                                  self.game_config)
        game.start_game()
