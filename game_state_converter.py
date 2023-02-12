import numpy as np

from game_environment.config_files.game_config import GameConfig
from game_environment.players.player import Player
from game_environment.trick import Trick
from game_environment.round import Round

import tensorflow as tf


def calculate_bidding_state_full_ohe(number_of_cards, playing_player, current_round: Round, total_number_of_rounds):
    own_hand = convert_hand_to_ohe(number_of_cards, playing_player.hand)
    trump_card = convert_trump_card_to_ohe(number_of_cards, current_round.trump_suit)
    round_number = determine_round_number(total_number_of_rounds, current_round.round_number)
    current_position_in_round = determine_player_pos_as_ohe(current_round.players, playing_player,
                                                            current_round.start_player)
    opponents_called_tricks = determine_players_called_tricks_as_ohe(current_round.players, playing_player,
                                                                     total_number_of_rounds)
    opponents_called_tricks = opponents_called_tricks.flatten()
    legal_moves = calculate_callable_tricks_to_ohe(total_number_of_rounds, current_round.round_number)
    observation = np.concatenate((own_hand, trump_card, round_number, current_position_in_round,
                                  opponents_called_tricks), axis=None)
    return {'observation': observation, 'legal_moves': legal_moves}


def calculate_bidding_state(number_of_cards, playing_player, current_round: Round
                            , total_number_of_rounds):
    own_hand = convert_hand_to_ohe(number_of_cards, playing_player.hand)
    trump_card = convert_trump_card_to_ohe(number_of_cards, current_round.trump_suit)
    current_position_in_round = determine_player_pos(current_round.players, playing_player, current_round.start_player)
    opponents_called_tricks, opponents_achieved_points = determine_players_called_tricks_and_points(
        current_round.players,
        playing_player)
    own_achieved_points = playing_player.points
    legal_moves = calculate_callable_tricks_to_ohe(total_number_of_rounds, current_round.round_number)
    observation = np.concatenate((own_hand, trump_card, current_round.round_number, current_position_in_round,
                                  opponents_called_tricks, own_achieved_points, opponents_achieved_points), axis=None)
    return {'observation': observation, 'legal_moves': legal_moves}


def calculate_playing_state(number_of_cards, playing_player: Player, current_trick: Trick, playable_cards):
    own_hand = convert_hand_to_ohe(number_of_cards, playing_player.hand)
    own_played_cards = convert_played_cards_to_ohe(number_of_cards, playing_player.played_cards)
    opponent_played_cards = []
    for player in current_trick.players:
        if player is not playing_player:
            opponent_played_cards.append(convert_played_cards_to_ohe(number_of_cards, player.played_cards))
    trump_card = convert_trump_card_to_ohe(number_of_cards, current_trick.trump_suit)
    current_position_in_round = determine_player_pos(current_trick.players, playing_player, current_trick.start_player)
    highest_card_in_trick = convert_highest_card_in_trick_to_ohe(number_of_cards, current_trick.highest_card_in_trick)
    cards_in_current_trick = convert_cards_in_trick_to_ohe(number_of_cards, current_trick.cards_in_trick)
    own_called_tricks = playing_player.guessed_tricks
    opponents_called_tricks, opponents_current_tricks, opponents_achieved_points = determine_players_called_and_current_tricks_and_points(
        current_trick.players, playing_player)
    own_achieved_points = playing_player.points
    legal_moves = calculate_playable_cards_to_ohe(playing_player.hand, playable_cards, number_of_cards)
    observation = np.concatenate((number_of_cards, own_hand, own_played_cards, opponent_played_cards, trump_card,
                                  highest_card_in_trick, cards_in_current_trick, current_trick.game_round_number,
                                  current_trick.trick_number,
                                  current_position_in_round,
                                  own_called_tricks, opponents_called_tricks,
                                  opponents_current_tricks, own_achieved_points,
                                  opponents_achieved_points), axis=None)

    return {'observation': observation, 'legal_moves': legal_moves}


def calculate_playing_state_small_input(number_of_cards, playing_player: Player, current_trick: Trick, playable_cards,
                                        number_of_colors):
    own_hand = convert_hand_to_ohe(number_of_cards, playing_player.hand)
    played_cards = []
    for player in current_trick.players:
        played_cards += player.played_cards
    all_played_cards = convert_played_cards_to_ohe(number_of_cards, played_cards)
    trump_color = convert_trump_card_to_color_ohe(number_of_colors, current_trick.trump_suit)
    current_position_in_round = determine_player_pos(current_trick.players, playing_player, current_trick.start_player)
    cards_in_current_trick = convert_cards_in_trick_to_ohe(number_of_cards, current_trick.cards_in_trick)
    own_called_tricks = playing_player.guessed_tricks
    legal_moves = calculate_playable_cards_to_ohe(playing_player.hand, playable_cards, number_of_cards)
    observation = np.concatenate((own_hand, all_played_cards, trump_color, cards_in_current_trick,
                                  current_trick.game_round_number, current_trick.trick_number,
                                  current_position_in_round,
                                  own_called_tricks, playing_player.won_tricks), axis=None)

    return {'observation': observation, 'legal_moves': legal_moves}


def calculate_playing_state_full_ohe(number_of_cards, playing_player: Player, current_trick: Trick,
                                     playable_cards,
                                     number_of_colors, total_number_of_rounds):
    own_hand = convert_hand_to_ohe(number_of_cards, playing_player.hand)
    played_cards = []
    for player in current_trick.players:
        played_cards += player.played_cards
    all_played_cards = convert_played_cards_to_ohe(number_of_cards, played_cards)
    trump_color = convert_trump_card_to_color_ohe(number_of_colors, current_trick.trump_suit)
    round_number = determine_round_number(total_number_of_rounds, current_trick.game_round_number)
    trick_number = determine_trick_number(total_number_of_rounds, current_trick.trick_number)
    current_position_in_round = determine_player_pos_as_ohe(current_trick.players, playing_player,
                                                            current_trick.start_player)
    cards_in_current_trick = convert_cards_in_trick_to_ohe(number_of_cards, current_trick.cards_in_trick)
    own_called_tricks = determine_tricks_as_ohe(playing_player.guessed_tricks, total_number_of_rounds)
    own_won_tricks = determine_tricks_as_ohe(playing_player.won_tricks, total_number_of_rounds)
    opponents_called_tricks = determine_players_called_tricks_as_ohe(current_trick.players, playing_player,
                                                                     total_number_of_rounds)
    opponents_called_tricks = opponents_called_tricks.flatten()
    opponents_won_tricks = determine_players_won_tricks_as_ohe(current_trick.players, playing_player,
                                                               total_number_of_rounds)
    opponents_won_tricks = opponents_won_tricks.flatten()
    legal_moves = calculate_playable_cards_to_ohe(playing_player.hand, playable_cards, number_of_cards)
    observation = np.concatenate((own_hand, all_played_cards, trump_color, cards_in_current_trick,
                                  round_number, trick_number, current_position_in_round, own_called_tricks,
                                  own_won_tricks, opponents_called_tricks, opponents_won_tricks),
                                 axis=None)

    return {'observation': observation, 'legal_moves': legal_moves}


def convert_cards_in_trick_to_ohe(number_of_cards, cards_in_trick):
    ohe_cards_in_trick = np.zeros(number_of_cards, dtype=np.int32)
    for card in cards_in_trick.values():
        ohe_cards_in_trick[card.card_id] = 1
    return ohe_cards_in_trick


def convert_highest_card_in_trick_to_ohe(number_of_cards, highest_card):
    ohe_highest_card = np.zeros(number_of_cards, dtype=np.int32)
    if highest_card is not None:
        ohe_highest_card[highest_card.card_id] = 1
    return ohe_highest_card


def convert_hand_to_ohe(number_of_cards, hand):
    ohe_hand = np.zeros(number_of_cards, dtype=np.int32)
    for card in hand:
        ohe_hand[card.card_id] = 1
    return ohe_hand


def convert_played_cards_to_ohe(number_of_cards, played_cards):
    ohe_played_cards = np.zeros(number_of_cards, dtype=np.int32)
    for card in played_cards:
        ohe_played_cards[card.card_id] = 1
    return ohe_played_cards


def convert_trump_card_to_ohe(number_of_cards, trump_suit):
    ohe_trump_card = np.zeros(number_of_cards, dtype=np.int32)
    ohe_trump_card[trump_suit.card_id] = 1
    return ohe_trump_card


def convert_trump_card_to_color_ohe(number_of_colors, trump_suit):
    ohe_trump_card = np.zeros(number_of_colors + 1, dtype=np.int32)
    ohe_trump_card[trump_suit.color_id] = 1
    return ohe_trump_card


def determine_player_pos_as_ohe(players, playing_player, start_player):
    player_pos_in_trick_ohe = np.zeros(len(players), dtype=np.int32)
    player_pos_in_trick = players.index(playing_player) - start_player
    if player_pos_in_trick >= 0:
        pos_in_round = player_pos_in_trick
    else:
        pos_in_round = len(players) - abs(player_pos_in_trick)
    player_pos_in_trick_ohe[pos_in_round] = 1
    return player_pos_in_trick_ohe


def determine_player_pos(players, playing_player, start_player):
    player_pos_in_trick = players.index(playing_player) - start_player
    if player_pos_in_trick >= 0:
        pos_in_round = player_pos_in_trick
    else:
        pos_in_round = len(players) - abs(player_pos_in_trick)
    return [pos_in_round]


def determine_players_called_tricks_as_ohe(players, playing_player, number_of_total_rounds):
    opponent_called_tricks = np.zeros(shape=(len(players) - 1, number_of_total_rounds + 1), dtype=np.int32)
    opponent_counter = 0
    for player in players:
        if player != playing_player:
            opponent_called_tricks[opponent_counter][player.guessed_tricks] = 1
    return opponent_called_tricks


def determine_players_won_tricks_as_ohe(players, playing_player, number_of_total_rounds):
    opponent_won_tricks = np.zeros(shape=(len(players) - 1, number_of_total_rounds + 1), dtype=np.int32)
    opponent_counter = 0
    for player in players:
        if player != playing_player:
            opponent_won_tricks[opponent_counter][player.won_tricks] = 1
    return opponent_won_tricks


def determine_tricks_as_ohe(trick_number, number_of_total_rounds):
    self_called_tricks = np.zeros(number_of_total_rounds + 1, dtype=np.int32)
    self_called_tricks[trick_number] = 1
    return self_called_tricks


def determine_players_called_tricks_and_points(players, playing_player):
    opponent_called_tricks = []
    opponent_achieved_points = []
    for player in players:
        if player != playing_player:
            opponent_called_tricks.append(player.guessed_tricks)
            opponent_achieved_points.append(player.points)
    return opponent_called_tricks, opponent_achieved_points


def determine_players_called_and_current_tricks_and_points(players, playing_player):
    opponent_called_tricks = []
    opponent_current_tricks = []
    opponent_achieved_points = []
    for player in players:
        if player != playing_player:
            opponent_called_tricks.append(player.guessed_tricks)
            opponent_current_tricks.append(player.won_tricks)
            opponent_achieved_points.append(player.points)
    return opponent_called_tricks, opponent_current_tricks, opponent_achieved_points


def calculate_playable_cards_to_ohe(hand, playable_cards, number_of_cards):
    ohe_playable_cards = np.zeros(number_of_cards, dtype=np.int32)
    for idx in playable_cards:
        ohe_playable_cards[hand[idx].card_id] = 1
    return ohe_playable_cards


def calculate_callable_tricks_to_ohe(total_number_of_rounds, current_round_number):
    callable_tricks = np.zeros((total_number_of_rounds + 1,), dtype=np.int32)
    for idx in range(current_round_number + 1):
        callable_tricks[idx] = 1
    return callable_tricks


def determine_round_number(total_number_of_rounds, current_round_number):
    round_number = np.zeros(total_number_of_rounds, dtype=np.int32)
    round_number[current_round_number - 1] = 1
    return round_number


def determine_trick_number(total_number_of_rounds, current_trick_number):
    trick_number = np.zeros(total_number_of_rounds, dtype=np.int32)
    trick_number[current_trick_number - 1] = 1
    return trick_number
