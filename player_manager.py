from game_environment.agent_wrappers.c51_bidding_wrapper import C51BiddingWrapper
from game_environment.agent_wrappers.c51_playing_wrapper import C51PlayingWrapper
from game_environment.agent_wrappers.dqn_bidding_wrapper import DQNBiddingWrapper
from game_environment.agent_wrappers.dqn_playing_wrapper import DQNPlayingWrapper
from game_environment.players.random_player import RandomPlayer
from game_environment.players.rl_player_bidding import RLPlayerBidding
from game_environment.players.rl_player_bidding_playing import RLPlayerBiddingPlaying
from game_environment.tf_environments.tf_wizard_bidding_env import TFWizardBiddingEnv
from game_environment.tf_environments.tf_wizard_play_env import TFWizardPlayEnv
from training_setup import TrainingSetup


class PlayerManager:
    def __init__(self, training_setup: TrainingSetup, directory):
        self.training_setup = training_setup
        self.setup_name = self.training_setup.name
        self.directory = directory

        self.game_config = self.training_setup.game_config
        self.training_config = self.training_setup.training_config

        self.agent_mode = self.training_setup.agent_mode
        self.player_mode = self.training_setup.player_mode

        if self.player_mode == "BidPlay":
            self.play_config = self.training_setup.play_config
        else:
            self.play_config = None

        self.bid_config = self.training_setup.bid_config

        self.players = []
        self.wrapper_list = []
        self.training_wrappers = []

        self.setup_agents()
        self.player = self.players[0]

    def setup_agents(self):
        if self.training_setup.agent_mode == "SingleAgent":
            self.setup_single_agent_training()
        elif self.training_setup.agent_mode == "MultiAgent":
            self.setup_multi_agent_training()

    def setup_pretrained_models(self):
        if self.training_setup.agent_mode == "SingleAgent":
            self.setup_pretrained_single_agent()
        elif self.training_setup.agent_mode == "MultiAgent":
            self.setup_pretrained_multi_agent()

    def setup_single_agent_training(self):
        if self.play_config is not None:
            self.setup_single_agent_global_variables(f'{self.bid_config.mode}_{self.play_config.mode}')
        else:
            self.setup_single_agent_global_variables(f'{self.bid_config.mode}')

        if self.bid_config.mode == "C51":
            self.setup_c51_bidding(self.players[0])

        if self.player_mode == "BidPlay" and self.play_config.mode == "C51":
            self.setup_c51_playing(self.players[0])

    def setup_multi_agent_training(self):
        self.setup_multi_agent_global_variables(f'{self.agent_mode}_{self.player_mode}')

        for idx, player in enumerate(self.players):
            if idx == 0:
                self.create_first_marl_agent(player)
            else:
                self.create_other_marl_agent(self.players[0], player)

    def setup_pretrained_single_agent(self):
        if self.play_config is not None:
            self.setup_single_agent_global_variables(f'{self.bid_config.mode}_{self.play_config.mode}')
        else:
            self.setup_single_agent_global_variables(f'{self.bid_config.mode}')

        if self.bid_config.mode == "C51":
            self.setup_c51_bidding(self.players[0])

        if self.player_mode == "BidPlay" and self.play_config.mode == "C51":
            self.setup_c51_playing(self.players[0])

    def setup_pretrained_multi_agent(self):
        self.setup_multi_agent_global_variables(f'{self.agent_mode}_{self.player_mode}')

    def setup_single_agent_global_variables(self, player_name):
        if self.player_mode == "BidPlay":
            player = RLPlayerBiddingPlaying(player_name, self.directory)
        else:
            player = RLPlayerBidding(player_name, self.directory)
        self.players.append(player)
        for x in range(self.game_config.num_players - 1):
            self.players.append(RandomPlayer('RndPlayer' + str(x + 1), self.directory))

    def setup_multi_agent_global_variables(self, player_name):
        for x in range(self.game_config.num_players):
            if self.player_mode == "BidPlay":
                player = RLPlayerBiddingPlaying(player_name + "_" + str(x + 1), self.directory)
            else:
                player = RLPlayerBidding(player_name + "_" + str(x + 1), self.directory)

            self.players.append(player)

    def create_first_marl_agent(self, player):

        if self.bid_config.mode == "C51":
            self.setup_c51_bidding(player)

        if self.player_mode == "BidPlay" and self.play_config.mode == "C51":
            self.setup_c51_playing(player)

    def create_other_marl_agent(self, first_player, this_player):
        if self.bid_config.mode == "C51":
            self.setup_c51_bidding_for_marl(first_player, this_player)

        if self.player_mode == "BidPlay" and self.play_config.mode == "C51":
            self.setup_c51_playing_for_marl(first_player, this_player)

    def setup_c51_bidding(self, player):
        train_bidding_py_env = TFWizardBiddingEnv(self.game_config, self.bid_config, player)
        eval_bidding_py_env = TFWizardBiddingEnv(self.game_config, self.bid_config, player)

        c51_bidding_wrapper = C51BiddingWrapper(player.name, train_bidding_py_env, eval_bidding_py_env, self.bid_config,
                                                self.game_config, self.agent_mode, self.directory,
                                                self.training_setup.existing_pretrained_model,
                                                self.training_setup.existing_bid_model_path)

        player.set_bidding_wrapper(c51_bidding_wrapper)
        self.wrapper_list.append(player.bidding_wrapper)
        self.training_wrappers.append(player.bidding_wrapper)

    def setup_c51_playing(self, player):

        train_playing_py_env = TFWizardPlayEnv(self.game_config, self.play_config, player)
        eval_playing_py_env = TFWizardPlayEnv(self.game_config, self.play_config, player)


        c51_playing_wrapper = C51PlayingWrapper(player.name, train_playing_py_env, eval_playing_py_env,
                                                self.play_config, self.game_config,
                                                self.agent_mode, self.directory,
                                                self.training_setup.existing_pretrained_model,
                                                self.training_setup.existing_play_model_path)

        player.set_playing_wrapper(c51_playing_wrapper)
        self.wrapper_list.append(player.playing_wrapper)
        self.training_wrappers.append(player.playing_wrapper)

    def setup_c51_bidding_for_marl(self, first_player, this_player):
        train_bidding_py_env = TFWizardBiddingEnv(self.game_config, self.bid_config, this_player)
        eval_bidding_py_env = TFWizardBiddingEnv(self.game_config, self.bid_config, this_player)

        c51_bidding_wrapper = C51BiddingWrapper(this_player.name, train_bidding_py_env, eval_bidding_py_env,
                                                self.bid_config, self.game_config, self.agent_mode, self.directory,
                                                self.training_setup.existing_pretrained_model,
                                                self.training_setup.existing_bid_model_path,
                                                first_player.bidding_wrapper.agent,
                                                first_player.bidding_wrapper.network,
                                                first_player.bidding_wrapper.replay_buffer,
                                                first_player.bidding_wrapper.dataset)

        this_player.set_bidding_wrapper(c51_bidding_wrapper)
        self.wrapper_list.append(this_player.bidding_wrapper)

    def setup_c51_playing_for_marl(self, first_player, this_player):
        train_playing_py_env = TFWizardPlayEnv(self.game_config, self.play_config, this_player)
        eval_playing_py_env = TFWizardPlayEnv(self.game_config, self.play_config, this_player)

        c51_playing_wrapper = C51PlayingWrapper(this_player.name, train_playing_py_env, eval_playing_py_env,
                                                self.play_config, self.game_config,
                                                self.agent_mode, self.directory,
                                                self.training_setup.existing_pretrained_model,
                                                self.training_setup.existing_play_model_path,
                                                first_player.playing_wrapper.agent,
                                                first_player.playing_wrapper.network,
                                                first_player.playing_wrapper.replay_buffer,
                                                first_player.playing_wrapper.dataset)

        this_player.set_playing_wrapper(c51_playing_wrapper)
        self.wrapper_list.append(this_player.playing_wrapper)

    @staticmethod
    def set_policies_to_best_policy(wrapper_list):
        best_bidding_wrapper = None
        best_playing_wrapper = None
        for wrapper in wrapper_list:
            if "Bidding" in wrapper.name:
                if best_bidding_wrapper is None or wrapper.total_avg_rewards[-1] > \
                        best_bidding_wrapper.total_avg_returns[-1]:
                    best_bidding_wrapper = wrapper
            if "Playing" in wrapper.name:
                if best_playing_wrapper is None or wrapper.total_avg_rewards[-1] > \
                        best_playing_wrapper.total_avg_returns[-1]:
                    best_playing_wrapper = wrapper

        for wrapper in wrapper_list:
            if "Bidding" in wrapper.name and wrapper is not best_bidding_wrapper:
                wrapper.set_policy(best_bidding_wrapper.policy)
            if "Playing" in wrapper.name and wrapper is not best_playing_wrapper:
                wrapper.set_policy(best_playing_wrapper.policy)

    def setup_dqn_bidding(self, player):
        train_bidding_py_env = TFWizardBiddingEnv(self.game_config, player)
        eval_bidding_py_env = TFWizardBiddingEnv(self.game_config, player)

        dqn_bidding_wrapper = DQNBiddingWrapper(train_bidding_py_env, eval_bidding_py_env, self.bid_config,
                                                self.agent_mode)

        player.set_bidding_wrapper(dqn_bidding_wrapper)
        self.wrapper_list.append(player.bidding_wrapper)
        self.training_wrappers.append(player.bidding_wrapper)

    def setup_dqn_playing(self, player):
        train_playing_py_env = TFWizardPlayEnv(self.game_config, player)
        eval_playing_py_env = TFWizardPlayEnv(self.game_config, player)

        dqn_playing_wrapper = DQNPlayingWrapper(train_playing_py_env, eval_playing_py_env, self.play_config,
                                                self.agent_mode)

        player.set_playing_wrapper(dqn_playing_wrapper)
        self.wrapper_list.append(player.playing_wrapper)
        self.training_wrappers.append(player.playing_wrapper)

    def setup_dqn_bidding_for_marl(self, first_player, this_player):
        train_bidding_py_env = TFWizardBiddingEnv(self.game_config, this_player)
        eval_bidding_py_env = TFWizardBiddingEnv(self.game_config, this_player)

        dqn_bidding_wrapper = DQNBiddingWrapper(train_bidding_py_env, eval_bidding_py_env, self.bid_config,
                                                self.agent_mode, first_player.bidding_wrapper.agent,
                                                first_player.bidding_wrapper.network,
                                                first_player.bidding_wrapper.replay_buffer,
                                                first_player.bidding_wrapper.dataset)

        this_player.set_bidding_wrapper(dqn_bidding_wrapper)
        self.wrapper_list.append(this_player.bidding_wrapper)

    def setup_dqn_playing_for_marl(self, first_player, this_player):
        train_playing_py_env = TFWizardPlayEnv(self.game_config, this_player)
        eval_playing_py_env = TFWizardPlayEnv(self.game_config, this_player)

        dqn_playing_wrapper = DQNPlayingWrapper(train_playing_py_env, eval_playing_py_env, self.play_config,
                                                self.agent_mode, first_player.playing_wrapper.agent,
                                                first_player.playing_wrapper.network,
                                                first_player.playing_wrapper.replay_buffer,
                                                first_player.playing_wrapper.dataset)

        this_player.set_playing_wrapper(dqn_playing_wrapper)
        self.wrapper_list.append(this_player.playing_wrapper)
