import os

from game_environment.config_files.c51_config import C51Config
from game_environment.config_files.game_config import GameConfig
from game_environment.config_files.rl_config import RLConfig
from game_environment.config_files.training_config import TrainingConfig


class TrainingSetup:

    def __init__(self, path, training_config="", game_config="", bid_config: RLConfig = "",
                 play_config: RLConfig = "", agent_mode="", player_mode="", existing_pretrained_model="",
                 existing_bid_model_path="", existing_play_model_path=""):
        if path == "":
            self.training_config = training_config
            self.game_config = game_config
            self.bid_config = bid_config
            self.play_config = play_config
            self.agent_mode = agent_mode
            self.player_mode = player_mode
            self.existing_pretrained_model = existing_pretrained_model
            self.existing_bid_model_path = existing_bid_model_path
            self.existing_play_model_path = existing_play_model_path
            self.name = ""
            if player_mode == "BidPlay":
                self.name = f'{agent_mode} {player_mode} {game_config.name} {bid_config.name} {play_config.name} {training_config.name}'
            else:
                self.name = f'{agent_mode} {player_mode} {game_config.name} {bid_config.name} {training_config.name}'
        else:
            self.load_setup_from_file(path)

    def load_setup_from_file(self, file_name):
        current_dir = os.getcwd()
        new_dir = os.path.join(current_dir, "pre_configs/setup")
        path = os.path.join(new_dir, file_name)
        with open(path, 'r') as f:
            lines = f.read().splitlines()
            self.name = lines[1]

            training_config_path = lines[3]
            game_config_path = lines[5]
            bid_config_path = lines[7]
            play_config_path = lines[9]

            self.training_config = TrainingConfig(training_config_path)
            self.game_config = GameConfig(game_config_path)
            self.bid_config = C51Config("bid/" + bid_config_path)
            if play_config_path != "":
                self.play_config = C51Config("play/" + play_config_path)

            self.agent_mode = lines[11]
            self.player_mode = lines[13]
            self.existing_pretrained_model = lines[15]
            if self.existing_pretrained_model == "True":
                self.existing_bid_model_path = lines[17]
                self.existing_play_model_path = lines[19]
            else:
                self.existing_bid_model_path = ""
                self.existing_play_model_path = ""

    def save_config_to_file(self, path):
        play_config_name = ""
        if self.player_mode == "BidPlay":
            play_config_name = self.play_config.name

        with open(os.path.join(path, f'{self.name}'), 'w') as f:
            f.write(f'Setup Name:\n'
                    f'{self.name}\n'
                    f'Training Config Name:\n'
                    f'{self.training_config.name}\n'
                    f'Game Config Name:\n'
                    f'{self.game_config.name}\n'
                    f'Bid Config Name:\n'
                    f'{self.bid_config.name}\n'
                    f'Play Config Name:\n'
                    f'{play_config_name}\n'
                    f'Agent Mode:\n'
                    f'{self.agent_mode}\n'
                    f'Player Mode:\n'
                    f'{self.player_mode}\n'
                    f'Existing Pre Trained Model:\n'
                    f'{self.existing_pretrained_model}\n'
                    f'Existing Bid Model Path:\n'
                    f'{self.existing_bid_model_path}\n'
                    f'Existing Play Model Path:\n'
                    f'{self.existing_play_model_path}\n'
                    )
        return
