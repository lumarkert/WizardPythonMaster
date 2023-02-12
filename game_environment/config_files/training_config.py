import os


class TrainingConfig:

    def __init__(self, path, name="", initial_collect_games="", collect_steps_per_iteration="", collect_games_interval="", num_iterations="",
                 log_interval="", num_eval_episodes="", eval_interval="", decay_interval=""):
        if path == "":
            self.name = name
            self.initial_collect_games = initial_collect_games
            self.collect_steps_per_iteration = collect_steps_per_iteration
            self.collect_games_interval = collect_games_interval
            self.num_iterations = num_iterations
            self.log_interval = log_interval
            self.num_eval_episodes = num_eval_episodes
            self.eval_interval = eval_interval
            self.decay_interval = decay_interval
        else:
            self.name = None
            self.initial_collect_games = None
            self.collect_steps_per_iteration = None
            self.collect_games_interval = None
            self.num_iterations = None
            self.log_interval = None
            self.num_eval_episodes = None
            self.eval_interval = None
            self.decay_interval = None
            self.load_config_from_file(path)

    def load_config_from_file(self, file_name):
        current_dir = os.getcwd()
        new_dir = os.path.join(current_dir, "pre_configs/training")
        path = os.path.join(new_dir, file_name)
        with open(path, 'r') as f:
            lines = f.read().splitlines()
            self.name = lines[1]
            self.initial_collect_games = int(lines[3])
            self.collect_steps_per_iteration = int(lines[5])
            self.collect_games_interval = int(lines[7])
            self.num_iterations = int(lines[9])
            self.log_interval = int(lines[11])
            self.num_eval_episodes = int(lines[13])
            self.eval_interval = int(lines[15])
            self.decay_interval = int(lines[17])

    def save_config_to_file(self, path):
        with open(os.path.join(path, f'{self.name}'), 'w') as f:
            f.write(f'Name: \n'
                    f'{self.name}\n'
                    f'Number of Initial Games:\n'
                    f'{self.initial_collect_games}\n'
                    f'Collected Steps per Iteration:\n'
                    f'{self.collect_steps_per_iteration}\n'
                    f'Collect Games every x Steps:\n'
                    f'{self.collect_games_interval}\n'
                    f'Number of Iterations:\n'
                    f'{self.num_iterations}\n'
                    f'Log Interval:\n'
                    f'{self.log_interval}\n'
                    f'Number of Evaluation Episodes:\n'
                    f'{self.num_eval_episodes}\n'
                    f'Evaluation Interval:\n'
                    f'{self.eval_interval}\n'
                    f'Learning Rate Decay Interval:\n'
                    f'{self.decay_interval}\n'
                    )
        return
