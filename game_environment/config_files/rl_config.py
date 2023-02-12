from abc import ABC, abstractmethod


class RLConfig(ABC):

    def __init__(self, mode):
        self.mode = mode

    @abstractmethod
    def load_config_from_file(self, path):
        return

    @abstractmethod
    def save_config_to_file(self, path):
        return
