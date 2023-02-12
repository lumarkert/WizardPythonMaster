from game_brettspielwelt.card import Card
from game_brettspielwelt.trick import Trick


class Round:

    def __init__(self, players, start_player, round_number, deck):
        self.players = players
        self.start_player = start_player
        self.round_number = round_number
        self.trump_suit = None
        self.deck = deck
        self.called_tricks = {}
        self.current_trick = None
        self.current_player = None
        self.deck.rebuild_deck()

        for player in players:
            self.called_tricks[player] = -1
            player.reset_played_cards()

    def deal_cards(self):
        print("Which Cards were dealt to Player? Please Type first the Color then the value like \"Y13\"")
        for card_number in range(self.round_number):
            print(f"Card Number {card_number + 1}")
            card_string = input("Enter the combination!")
            card_id = self.determine_card(card_string)
            while card_id is None:
                card_string = input("Please enter a valid combination!")
                card_id = self.determine_card(card_string)
            card = self.deck.cards[int(card_id)]
            self.players[0].add_card_to_hand(card)
            print(f"Card {card.value} of {card.color} was added")

        cards_correct = False
        while not cards_correct:
            self.players[0].print_hand()
            cards_correct_input = input(
                "Are these Cards in your hand correct? Type yes to confirm or enter a index to tell which card is wrong.")
            if cards_correct_input == "yes" or cards_correct_input == "Yes":
                print("The cards in hand were confirmed.")
                cards_correct = True
            elif cards_correct_input.isdigit() and (int(cards_correct_input) - 1) < len(self.players[0].hand):
                card_string = input("Enter the combination!")
                card_id = self.determine_card(card_string)
                while card_id is None:
                    card_string = input("Please enter a valid combination!")
                    card_id = self.determine_card(card_string)
                card = self.deck.cards[int(card_id)]
                self.players[0].hand[int(cards_correct_input) - 1] = card
            else:
                print("Please enter a valid input!")

        print("-----------------------------------")
        self.players[0].count_hand_by_color(self.deck)

    def determine_card(self, card_string):
        if len(card_string) < 2:
            return None
        color = card_string[0]
        if not color.isalpha() or (color != "y" and color != "r" and color != "b" and color != "g" and color != "w" and color != "j"):
            return None
        if len(card_string) == 3:
            value = card_string[1:3]
        else:
            value = card_string[1]
        if (color == "w" or color == "j") and int(value) > self.deck.number_of_wizards:
            return None
        if not value.isdigit() or int(value) > self.deck.number_of_values:
            return None
        value = int(value)
        card_start_idx = 0
        if color == "g":
            card_start_idx = 0
        elif color == "b":
            card_start_idx = 13
        elif color == "y":
            card_start_idx = 26
        elif color == "r":
            card_start_idx = 39
        elif color == "w":
            card_start_idx = 52
        elif color == "j":
            card_start_idx = 56

        return card_start_idx + value - 1

    def reveal_trump(self):
        print("Please specify the trump card!")
        card_string = input("Enter the combination!")
        card_id = self.determine_card(card_string)
        while card_id is None:
            card_string = input("Please enter a valid combination!")
            card_id = self.determine_card(card_string)
        self.trump_suit = self.deck.cards[int(card_id)]

        card_correct = False
        while not card_correct:
            print(f"The Trump Card is the {self.trump_suit.value} of {self.trump_suit.color}.")
            cards_correct_input = input("Is this card correct? Type yes to confirm or no to choose again.")
            if cards_correct_input == "yes" or cards_correct_input == "Yes":
                print("The trump card is confirmed.")
                card_correct = True
            elif cards_correct_input == "no" or cards_correct_input == "No":
                card_string = input("Enter the combination!")
                card_id = self.determine_card(card_string)
                while card_id is None:
                    card_string = input("Please enter a valid combination!")
                    card_id = self.determine_card(card_string)
            else:
                print("Please enter a valid input!")

        if self.trump_suit.value == 15:
            self.trump_suit.color = self.players[self.start_player].set_trump_suit(self.deck)
            print(f"Player {self.players[self.start_player].name} picked the color {self.trump_suit.color}!")

        print("-----------------------")

    def call_trick_from_player(self, calling_player):
        number_of_tricks_called = self.players[calling_player].call_tricks(self)
        self.called_tricks[self.players[calling_player]] = number_of_tricks_called
        print(f"Player {self.players[calling_player].name} called {number_of_tricks_called} Tricks!")

    def print_called_tricks(self):
        print('These are the called Tricks for this round:')
        for player in self.called_tricks:
            if self.called_tricks[player] != -1:
                print(f'{player.name} called {self.called_tricks[player]} tricks!')
        print('')

    def call_tricks(self):
        for i in range(len(self.players)):
            calling_player = (self.start_player + i) % len(self.players)
            self.call_trick_from_player(calling_player)
        self.print_called_tricks()

    def setup_new_round(self):
        self.deal_cards()
        self.reveal_trump()

    def play_trick(self, trick_number):
        self.current_trick = Trick(self.players, self.start_player, self.trump_suit, trick_number + 1, self.round_number, self.deck)
        player_won = self.current_trick.play_trick()
        player_won.increment_won_tricks()
        print(f'{player_won.name} won trick number {trick_number + 1}')
        if trick_number + 1 != self.round_number:
            print(f'{player_won.name} will start the next trick')
        self.start_player = self.players.index(player_won)
        player_won = None
        print('-----------------------------------------------')

    def print_round_overview(self):
        print('Round Overview :')
        for player in self.players:
            print("\033[4m" + player.name + "\033[0m")
            print(f'Called Tricks: {player.guessed_tricks}')
            print(f'Won Tricks: {player.won_tricks}')
            player.score_points()
            player.wrap_up_round(self.current_trick)
            print('')

    def print_current_scores(self):
        print('Current Scores:')
        for player in self.players:
            print(f'{player.name}: {player.points}')
        print('-----------------------------------------------')

    def play_round(self):
        self.setup_new_round()
        self.call_tricks()
        for j in range(self.round_number):
            self.play_trick(j)
        self.print_round_overview()




