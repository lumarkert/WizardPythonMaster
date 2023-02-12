from game_environment.card import Card
from game_environment.trick import Trick


class Round:

    def __init__(self, players, start_player, round_number, deck):
        self.players = players
        self.start_player = start_player
        self.round_number = round_number
        self.deck = deck
        self.trump_suit = None
        self.called_tricks = {}
        self.current_trick = None
        self.current_player = None
        for player in players:
            self.called_tricks[player] = -1
            player.reset_played_cards()

    def deal_cards(self):
        for _ in range(self.round_number):
            for player in self.players:
                player.add_card_to_hand(self.deck.draw_card())
        for player in self.players:
            player.count_hand_by_color(self.deck)

    def reveal_trump(self):
        if len(self.deck.cards) != 0:
            self.trump_suit = self.deck.draw_card()
            print(f'The revealed card that specifies the trump suit is {self.trump_suit}')
            if self.trump_suit.value == 15:
                self.trump_suit.color = self.players[self.start_player].set_trump_suit(self.deck)
            print(f'The Trump Suit of Round {self.round_number} is {self.trump_suit.color}\n')
            if self.trump_suit.value == 14:
                print(f'A Jester was pulled as trump suit card! Therefore there is no trump suit this round!')
        else:
            self.trump_suit = Card("NOC", 14, self.deck.number_of_cards - 1, 0)
            print(f'It is the last Round! There is NO TRUMP SUIT!')

    def call_trick_from_player(self, calling_player):
        self.called_tricks[self.players[calling_player]] = self.players[calling_player].call_tricks(self)

    def print_called_tricks(self):
        print('These are the called Tricks for this round:')
        for player in self.called_tricks:
            if self.called_tricks[player] != -1:
                print(f'{player} called {self.called_tricks[player]} tricks!')
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
        self.current_trick = Trick(self.players, self.start_player, self.trump_suit, trick_number + 1, self.round_number)
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
            player.set_round_ended()
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




