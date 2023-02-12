import numbers
import os

from game_environment.config_files.rl_config import RLConfig


class C51Config(RLConfig):

    def __init__(self, path, name="", replay_buffer_capacity="", fc_layers_number="", fc_layer_1="", fc_layer_2="",
                 fc_layer_3="", batch_size="", learning_rate="", allow_lr_decay="", lr_decay_rate="",
                 lr_decay_threshold="",
                 min_lr="", gamma="", num_atoms="", min_q_value="", max_q_value="", n_step_update="", epsilon_greedy="",
                 allow_epsilon_decay="", epsilon_decay_rate="", epsilon_decay_threshold="", min_epsilon="",
                 reward_function="", playing_environment=""):
        super().__init__("C51")
        if path == "":
            self.name = name
            self.replay_buffer_capacity = replay_buffer_capacity
            self.fc_layers_number = fc_layers_number
            self.fc_layer_1 = fc_layer_1
            self.fc_layer_2 = fc_layer_2
            self.fc_layer_3 = fc_layer_3
            self.batch_size = batch_size
            self.learning_rate = learning_rate
            self.allow_lr_decay = allow_lr_decay
            self.lr_decay_rate = lr_decay_rate
            self.lr_decay_threshold = lr_decay_threshold
            self.min_lr = min_lr
            self.gamma = gamma
            self.num_atoms = num_atoms
            self.min_q_value = min_q_value
            self.max_q_value = max_q_value
            self.n_step_update = n_step_update
            self.epsilon_greedy = epsilon_greedy
            self.allow_epsilon_decay = allow_epsilon_decay
            self.epsilon_decay_rate = epsilon_decay_rate
            self.epsilon_decay_threshold = epsilon_decay_threshold
            self.min_epsilon = min_epsilon
            self.reward_function = reward_function
            self.environment = playing_environment
            if self.min_q_value == "Auto":
                self.min_q_value_auto = True
            else:
                self.min_q_value_auto = False
            if self.max_q_value == "Auto":
                self.max_q_value_auto = True
            else:
                self.max_q_value_auto = False
        else:
            self.name = None
            self.replay_buffer_capacity = None
            self.fc_layers_number = None
            self.fc_layer_1 = None
            self.fc_layer_2 = None
            self.fc_layer_3 = None
            self.batch_size = None
            self.learning_rate = None
            self.allow_lr_decay = None
            self.lr_decay_rate = None
            self.lr_decay_threshold = None
            self.min_lr = None
            self.gamma = None
            self.num_atoms = None
            self.min_q_value = None
            self.min_q_value_auto = None
            self.max_q_value = None
            self.max_q_value_auto = None
            self.n_step_update = None
            self.epsilon_greedy = None
            self.allow_epsilon_decay = None
            self.epsilon_decay_rate = None
            self.epsilon_decay_threshold = None
            self.min_epsilon = None
            self.reward_function = None
            self.environment = None
            self.load_config_from_file(path)

    def create_fc_layer_variable(self):
        if self.fc_layers_number == 8:
            return [64, 64, 64, 64, 64, 64, 64, 64]
        elif self.fc_layers_number == 3:
            return [self.fc_layer_1, self.fc_layer_2, self.fc_layer_3]
        elif self.fc_layers_number == 2:
            return [self.fc_layer_1, self.fc_layer_2]
        else:
            return [self.fc_layer_1]

    def load_config_from_file(self, file_name):
        current_dir = os.getcwd()
        new_dir = os.path.join(current_dir, "pre_configs/c51")
        path = os.path.join(new_dir, file_name)
        with open(path, 'r') as f:
            lines = f.read().splitlines()
            self.name = lines[1]
            self.replay_buffer_capacity = int(lines[3])
            self.fc_layers_number = int(lines[5])
            self.fc_layer_1 = int(lines[7])
            self.fc_layer_2 = int(lines[9])
            self.fc_layer_3 = int(lines[11])
            self.batch_size = int(lines[13])
            self.learning_rate = float(lines[15])
            self.allow_lr_decay = lines[17]
            self.lr_decay_rate = float(lines[19])
            self.lr_decay_threshold = int(lines[21])
            self.min_lr = float(lines[23])
            self.gamma = float(lines[25])
            self.num_atoms = int(lines[27])
            self.min_q_value = lines[29]
            if self.min_q_value == "Auto":
                self.min_q_value_auto = True
            else:
                self.min_q_value_auto = False
                self.min_q_value = int(self.min_q_value)

            self.max_q_value = lines[31]
            if self.max_q_value == "Auto":
                self.max_q_value_auto = True
            else:
                self.max_q_value_auto = False
                self.max_q_value = int(self.max_q_value)

            self.n_step_update = int(lines[33])

            self.epsilon_greedy = float(lines[35])
            self.allow_epsilon_decay = lines[37]
            self.epsilon_decay_rate = float(lines[39])
            self.epsilon_decay_threshold = int(lines[41])
            self.min_epsilon = float(lines[43])

            self.reward_function = lines[45]
            self.environment = lines[47]

    def save_config_to_file(self, path, mode):
        with open(os.path.join(path, f'{mode} {self.name}'), 'w') as f:
            f.write(f'Name:\n'
                    f'{self.name}\n'
                    f'Replay-Buffer Capacity:\n'
                    f'{self.replay_buffer_capacity}\n'
                    f'Total FC Layers:\n'
                    f'{self.fc_layers_number}\n'
                    f'Layer 1 Length:\n'
                    f'{self.fc_layer_1}\n'
                    f'Layer 2 Length:\n'
                    f'{self.fc_layer_2}\n'
                    f'Layer 3 Length:\n'
                    f'{self.fc_layer_3}\n'
                    f'Batch Size:\n'
                    f'{self.batch_size}\n'
                    f'Learning Rate:\n'
                    f'{self.learning_rate}\n'
                    f'Allow Learning Rate Decay:\n'
                    f'{self.allow_lr_decay}\n'
                    f'Learning Rate Decay Rate:\n'
                    f'{self.lr_decay_rate}\n'
                    f'Learning Rate Decay Threshold:\n'
                    f'{self.lr_decay_threshold}\n'
                    f'Minimum Learning Rate:\n'
                    f'{self.min_lr}\n'
                    f'Gamma:\n'
                    f'{self.gamma}\n'
                    f'Number of Atoms:\n'
                    f'{self.num_atoms}\n'
                    f'Minimal Q Value:\n'
                    f'{self.min_q_value}\n'
                    f'Maximal Q Value:\n'
                    f'{self.max_q_value}\n'
                    f'Update every n Steps:\n'
                    f'{self.n_step_update}\n'
                    f'Epsilon Greedy:\n'
                    f'{self.epsilon_greedy}\n'
                    f'Allow Epsilon Decay:\n'
                    f'{self.allow_epsilon_decay}\n'
                    f'Epsilon Decay Rate:\n'
                    f'{self.epsilon_decay_rate}\n'
                    f'Epsilon Decay Threshold:\n'
                    f'{self.epsilon_decay_threshold}\n'
                    f'Minimum Epsilon:\n'
                    f'{self.min_epsilon}\n'
                    f'Reward Function:\n'
                    f'{self.reward_function}\n'
                    f'Environment:\n'
                    f'{self.environment}\n'
                    )
        return
