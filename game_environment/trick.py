class Trick:
    def __init__(self, players, start_player, trump_suit, trick_number, game_round_number):
        self.cards_in_trick = {}
        self.trick_suit = None
        self.players = players
        self.start_player = start_player
        self.highest_card_in_trick = None
        self.trump_suit = trump_suit
        self.winning_player = None
        self.trick_number = trick_number
        self.game_round_number = game_round_number

    def player_plays_card(self, current_player):
        return self.players[current_player].play_card(self)

    def review_played_card(self, played_card, current_player):
        if self.trick_suit is None and played_card.value <= 13:
            self.trick_suit = played_card.color
        if self.check_if_card_tops(played_card):
            self.winning_player = self.players[current_player]

    def add_card_to_trick(self, played_card, current_player):
        self.cards_in_trick[self.players[current_player]] = played_card

    def print_cards_in_trick(self):
        print('')
        print(f'The played cards of this trick (trump suit is {self.trump_suit.color}) are:')
        for card_in_trick in self.cards_in_trick:
            print(f'{card_in_trick.name} played this card: {self.cards_in_trick[card_in_trick]}')
        print('')

    def determine_current_player(self, i):
        return (i + self.start_player) % len(self.players)

    def play_trick(self):
        for i in range(len(self.players)):
            current_player = self.determine_current_player(i)
            played_card = self.player_plays_card(current_player)
            self.review_played_card(played_card, current_player)
            self.players[current_player].add_card_to_played_cards(played_card)
            self.add_card_to_trick(played_card, current_player)
        self.print_cards_in_trick()
        return self.winning_player

    def check_if_card_tops(self, card):
        if self.highest_card_in_trick is None:
            self.highest_card_in_trick = card
            return True
        if self.highest_card_in_trick.value == 15:
            return False
        if self.highest_card_in_trick.value == 14 and card.value != 14:
            return True
        if card.value == 15:
            self.highest_card_in_trick = card
            return True
        if self.trump_suit.color != "NOC" and self.highest_card_in_trick.color != self.trump_suit.color and card.color == self.trump_suit.color :
            self.highest_card_in_trick = card
            return True
        if card.color == self.highest_card_in_trick.color and card.value > self.highest_card_in_trick.value:
            self.highest_card_in_trick = card
            return True
        else:
            return False
