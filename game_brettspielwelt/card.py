class Card:
    def __init__(self, color, value, card_id, color_id):
        # Colors: NOC = no color, GRE = green, RED = red, YEL = yellow, BLU = blue
        self.color = color
        self.color_id = color_id
        # Values match the numbers on the cards, 14 = jester, 15 = wizard
        self.value = value
        self.card_id = card_id

    def __str__(self):
        if self.value == 14:
            return "Jester"
        if self.value == 15:
            return "Wizard"
        else:
            return f"{self.value} of {self.color}"
