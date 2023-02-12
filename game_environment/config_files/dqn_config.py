import os

from game_environment.config_files.rl_config import RLConfig


class DQNConfig(RLConfig):

    def __init__(self, replay_buffer_capacity, fc_layer_params, batch_size, learning_rate, gamma, n_step_update,
                 epsilon_greedy, bias_init_constant, random_uniform_min, random_uniform_max, path=""):
        super().__init__("DQN")
        if path == "":
            self.replay_buffer_capacity = replay_buffer_capacity
            self.fc_layer_params = fc_layer_params
            self.batch_size = batch_size
            self.learning_rate = learning_rate
            self.gamma = gamma
            self.n_step_update = n_step_update
            self.epsilon_greedy = epsilon_greedy
            self.bias_init_constant = bias_init_constant
            self.random_uniform_min = random_uniform_min
            self.random_uniform_max = random_uniform_max
        else:
            self.replay_buffer_capacity = None
            self.fc_layer_params = None
            self.batch_size = None
            self.learning_rate = None
            self.gamma = None
            self.n_step_update = None
            self.epsilon_greedy = None
            self.bias_init_constant = None
            self.random_uniform_min = None
            self.random_uniform_max = None
            self.load_config_from_file(path)

    def load_config_from_file(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()
            self.replay_buffer_capacity = lines[1]
            self.fc_layer_params = lines[3]
            self.batch_size = lines[5]
            self.learning_rate = lines[7]
            self.gamma = lines[9]
            self.n_step_update = lines[11]
            self.epsilon_greedy = lines[13]
            self.bias_init_constant = lines[15]
            self.random_uniform_min = lines[17]
            self.random_uniform_max = lines[19]

    def save_config_to_file(self, path):
        with open(os.path.join(path, "c51_config"), 'w') as f:
            f.write(f'Replay-Buffer Capacity: \n'
                    f'{self.replay_buffer_capacity} \n'
                    f'FC Layer Parameters : \n'
                    f'{self.fc_layer_params} \n'
                    f'Batch Size: \n'
                    f'{self.batch_size} \n'
                    f'Learning Rate: \n'
                    f'{self.learning_rate} \n'
                    f'Gamma: \n'
                    f'{self.gamma} \n'
                    f'Update every n Steps: \n'
                    f'{self.n_step_update} \n'
                    f'Epsilon Greedy: \n'
                    f'{self.epsilon_greedy} \n'
                    f'bias_init_constant: \n'
                    f'{self.bias_init_constant} \n'
                    f'random_uniform_min: \n'
                    f'{self.random_uniform_min} \n'
                    f'random_uniform_max: \n'
                    f'{self.random_uniform_max} \n'
                    )
        return
