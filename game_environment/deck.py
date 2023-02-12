import random
from game_environment.card import Card



class Deck:
    def __init__(self, colors=["GRE", "BLU", "YEL", "RED"], number_of_values=13, number_of_wizards=4,
                 number_of_jesters=4):
        self.cards = []
        self.colors = colors
        self.number_of_values = number_of_values
        self.number_of_wizards = number_of_wizards
        self.number_of_jesters = number_of_jesters
        self.number_of_cards = (len(colors) * number_of_values) + number_of_jesters + number_of_wizards

    def rebuild_deck(self):
        self.cards = []
        card_id = 0
        for idx, color in enumerate(self.colors):
            for value in range(1, self.number_of_values + 1):
                self.cards.append(Card(color, value, card_id, idx + 1))
                card_id += 1
        for _ in range(self.number_of_wizards):
            self.cards.append(Card("NOC", 15, card_id, 0))
            card_id += 1
        for _ in range(self.number_of_jesters):
            self.cards.append(Card("NOC", 14, card_id, 0))
            card_id += 1

    # Shuffles the Card Deck randomly. The deck is shuffled the same way if the given seed is the same. If the seed
    # is not passed to the function, or is set to -1 the deck is completely randomly shuffled.
    def shuffle(self, seed=-1):
        if seed <= -1:
            random.seed()
            random.shuffle(self.cards)
        else:
            random.seed(seed)
            random.shuffle(self.cards)

    def draw_card(self):
        if len(self.cards) != 0:
            card = self.cards.pop()
            return card
        else:
            return None

