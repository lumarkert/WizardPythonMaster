import os
import sys
import time

from game_environment.deck import Deck
from game_environment.game import Game
from game_environment.players.random_player import RandomPlayer
from game_environment.players.real_player import RealPlayer
import matplotlib.pyplot as plt


def main():
    #sys.stdout = open(os.devnull, 'w')
    deck = create_deck()
    players = []
    players.append(create_rnd_player('RndPlayer1'))
    players.append(create_rnd_player('RndPlayer2'))
    players.append(create_rnd_player('RndPlayer3'))
    scores = []
    average_points_per_round = []
    for i in range(len(players)):
        scores.append([])
        average_points_per_round.append([])
        average_points_per_round[i] = [0 for x in range(20)]

    number_of_games = 100

    start_time = time.time()
    for i in range(number_of_games):
        game = create_game(players, deck)
        game.start_game()
        for j in range(len(players)):
            scores[j].append(players[j].points)
            for idx, points_in_round in enumerate(players[j].points_per_round):
                average_points_per_round[j][idx] += players[j].points_per_round[idx]
            players[j].reset_game_stats()

    sys.stdout = sys.__stdout__
    print(f'Total Execution Time: {time.time() - start_time}')
    calculate_statistic(players, scores, average_points_per_round, number_of_games)


def calculate_statistic(players, scores, average_points_per_round, number_of_games):
    total_average = 0
    for i in range(len(players)):
        print('')
        print(f' {players[i].name} lowest Score was: {min(scores[i])}')
        print(f' {players[i].name} highest Score was: {max(scores[i])}')

        average = sum(scores[i]) / len(scores[i])
        total_average += average
        print(f' {players[i].name} average Score is: {average}')

        average_accuracy = sum(players[i].history_accuracy) / len(players[i].history_accuracy)
        print(f'Average Accuracy of told Tricks is {average_accuracy}')

        for idx, entrance in enumerate(average_points_per_round[i]):
            average_points_per_round[i][idx] = entrance / number_of_games
        print(average_points_per_round[i])

        plt.xticks(range(1,21))
        plt.plot(range(1,21), average_points_per_round[i])
        plt.grid()
        plt.savefig(players[i].name +".png")
        print(sum(average_points_per_round[i]))
    total_average = total_average / len(scores)
    print(f' Total average Score is: {total_average}')




def create_deck():
    deck = Deck()
    deck.rebuild_deck()
    deck.shuffle()
    return deck


def create_rnd_player(name):
    player = RandomPlayer(name)
    return player


def create_real_player(name):
    player = RealPlayer(name)
    return player


def create_game(players, deck):
    return Game(players, deck)


main()
