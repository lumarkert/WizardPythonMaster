import os


class GameConfig:

    def __init__(self, path, name="", num_players="", number_of_colors="", number_of_values="", number_of_wizards="",
                 number_of_jesters="", number_of_total_rounds=""):

        if path == "":
            self.name = name
            self.num_players = num_players
            self.number_of_colors = number_of_colors
            self.number_of_values = number_of_values
            self.number_of_wizards = number_of_wizards
            self.number_of_jesters = number_of_jesters
            self.number_of_total_rounds = number_of_total_rounds

        else:
            self.name = None
            self.num_players = None
            self.number_of_colors = None
            self.number_of_values = None
            self.number_of_wizards = None
            self.number_of_jesters = None
            self.number_of_total_rounds = None
            self.number_of_total_cards = None
            self.bid_steps_per_game = None
            self.play_steps_per_game = None
            self.load_config_from_file(path)

        self.number_of_total_cards = self.number_of_colors * self.number_of_values + self.number_of_wizards + self.number_of_jesters
        self.bid_steps_per_game = self.number_of_total_rounds * self.num_players
        self.play_steps_per_game = (self.number_of_total_rounds * (self.number_of_total_rounds + 1) / 2) * self.num_players
        self.colors = self.create_colors_variable()
        self.max_points = self.number_of_total_rounds * 10 + 20
        self.min_points = self.number_of_total_rounds * - 10

    def create_colors_variable(self):
        if self.number_of_colors == 1:
            return ["GRE"]
        elif self.number_of_colors == 2:
            return ["GRE", "BLU"]
        elif self.number_of_colors == 3:
            return ["GRE", "BLU", "YEL"]
        elif self.number_of_colors == 4:
            return ["GRE", "BLU", "YEL", "RED"]

    def load_config_from_file(self, file_name):
        current_dir = os.getcwd()
        new_dir = os.path.join(current_dir, "pre_configs/game")
        path = os.path.join(new_dir, file_name)
        with open(path, 'r') as f:
            lines = f.read().splitlines()
            self.name = lines[1]
            self.num_players = int(lines[3])
            self.number_of_colors = int(lines[5])
            self.number_of_values = int(lines[7])
            self.number_of_wizards = int(lines[9])
            self.number_of_jesters = int(lines[11])
            self.number_of_total_rounds = int(lines[13])

    def save_config_to_file(self, path):
        with open(os.path.join(path, f"{self.name}"), 'w') as f:
            f.write(f'Name:\n'
                    f'{self.name}\n'
                    f'Number of Players:\n'
                    f'{self.num_players}\n'
                    f'Colors in Game:\n'
                    f'{self.number_of_colors}\n'
                    f'Number of Values:\n'
                    f'{self.number_of_values}\n'
                    f'Number of Wizards:\n'
                    f'{self.number_of_wizards}\n'
                    f'Number of Jesters:\n'
                    f'{self.number_of_jesters}\n'
                    f'Number of Total Rounds:\n'
                    f'{self.number_of_total_rounds}\n'
                    )
        return
