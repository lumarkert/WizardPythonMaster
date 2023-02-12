import os
import csv
from datetime import datetime
from itertools import zip_longest

import numpy as np
import time as t
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator


def save_files(directory, start_step, wrapper_list, training_wrappers, training_setup, bid_config, play_config,
               game_config, training_config,
               players, start_time):
    save_config(directory, training_setup, bid_config, play_config, game_config, training_config)
    plot_avg_winrate_for_players(directory, start_step, training_config, players)
    plot_avg_score_for_players(directory, start_step, training_config, players)
    plot_avg_accuracy_for_players(directory, start_step, training_config, players)
    plot_avg_reward_all_wrappers(directory, start_step, wrapper_list, training_config)
    plot_avg_return_all_wrappers(directory, start_step, wrapper_list, training_config)
    plot_stats_per_round_for_last_round(start_step, game_config, training_config, players)
    save_player_data(start_step, training_config, players, game_config)
    for wrapper in wrapper_list:
        if training_wrappers.count(wrapper) > 0:
            plot_train_loss(wrapper, start_step, training_config)
            plot_replay_buffer_size(wrapper, start_step, training_config)
            plot_learning_rate_decay(wrapper, start_step, training_config)
            plot_epsilon_decay(wrapper, start_step, training_config)
            wrapper.save_check_pointer()

        plot_avg_reward(wrapper, start_step, training_config)
        plot_avg_return(wrapper, start_step, training_config)
        save_final_stats(wrapper, start_time)
        save_wrapper_data(wrapper, start_step, training_config)


def plot_stats_per_round_for_last_round(start_step, game_config, training_config, players):
    game_steps = []
    for i in range(int(game_config.number_of_total_rounds)):
        game_steps.append(i + 1)
    plot_steps = [int(start_step) + 1, int(start_step) + int(training_config.num_iterations / 2),
                  int(start_step) + int(training_config.num_iterations)]
    for player in players:
        for plot_idx, plot_step in enumerate(plot_steps):

            plt.plot(game_steps, player.avg_accuracy_per_round[plot_idx], label=player.name)
            plt.xticks(game_steps)
            plt.ylabel('Average Accuracy')
            plt.xlabel('Round Number')
            plt.ylim(top=105, bottom=-5)
            plt.grid()
            plt.axhline(y=np.nanmean(player.avg_accuracy_per_round[plot_idx]))
            file_name = f'per_round/avg_accuracy_per_round_step{plot_step}_{player.name}.png'
            plt.legend()
            plt.savefig(os.path.join(player.player_directory, file_name))
            plt.clf()

            plt.plot(game_steps, player.avg_points_per_round[plot_idx], label=player.name)
            plt.xticks(game_steps)
            plt.ylabel('Average Points')
            plt.xlabel('Round Number')
            plt.ylim(top=50, bottom=-50)
            plt.axhline(y=np.nanmean(player.avg_points_per_round[plot_idx]))
            plt.grid()
            file_name = f'per_round/avg_points_per_round_step{plot_step}_{player.name}.png'
            plt.legend()
            plt.savefig(os.path.join(player.player_directory, file_name))
            plt.clf()

            plt.boxplot(player.history_guessed_tricks_sorted_after_round_number[plot_idx])
            plt.ylabel('Guessed Tricks')
            plt.xlabel('Round Number')
            plt.ylim(top=(game_config.number_of_total_rounds+1), bottom=-1)
            file_name = f'per_round/guessed_tricks_step{plot_step}_{player.name}.png'
            plt.savefig(os.path.join(player.player_directory, file_name))
            plt.clf()

            plt.boxplot(player.history_points_sorted_after_round_number[plot_idx])
            plt.ylabel('Points')
            plt.xlabel('Round Number')
            plt.ylim(top=(game_config.max_points + 10), bottom=(game_config.min_points - 10))
            file_name = f'per_round/boxplot_points_step{plot_step}_{player.name}.png'
            plt.savefig(os.path.join(player.player_directory, file_name))
            plt.clf()


def plot_learning_rate_decay(wrapper, start_step, training_config):
    steps = range(start_step, start_step + training_config.num_iterations + 1, training_config.decay_interval)
    plt.plot(steps, wrapper.history_learning_rates)
    plt.ylabel('Learning Rate')
    plt.xlabel('Step')
    plt.ylim(bottom=0)
    file_name = f'learning_rate_decay.png'
    plt.savefig(os.path.join(wrapper.wrapper_directory, file_name))
    plt.clf()


def plot_epsilon_decay(wrapper, start_step, training_config):
    steps = range(start_step, start_step + training_config.num_iterations + 1, training_config.decay_interval)
    plt.plot(steps, wrapper.history_epsilons)
    plt.ylabel('Epsilon')
    plt.xlabel('Step')
    plt.ylim(bottom=0)
    file_name = f'epsilon_decay.png'
    plt.savefig(os.path.join(wrapper.wrapper_directory, file_name))
    plt.clf()


def save_wrapper_data(wrapper, start_step, training_config):
    steps = range(start_step, start_step + training_config.num_iterations + 1, training_config.eval_interval)
    data_dir = os.path.join(wrapper.wrapper_directory, "data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    data = [steps, wrapper.total_avg_rewards]
    export_data = zip_longest(*data, fillvalue='')

    with open(os.path.join(data_dir, f'avg_rewards_csv_{wrapper.name}.csv'), 'w', encoding="ISO-8859-1",
              newline='') as f:
        write = csv.writer(f)
        write.writerow(['Step', 'Average Rewards'])
        write.writerows(export_data)

    data = [steps, wrapper.total_avg_returns]
    export_data = zip_longest(*data, fillvalue='')

    with open(os.path.join(data_dir, f'avg_returns_csv_{wrapper.name}.csv'), 'w', encoding="ISO-8859-1",
              newline='') as f:
        write = csv.writer(f)
        write.writerow(['Step', 'Average Returns'])
        write.writerows(export_data)

    steps = range(0, training_config.num_iterations, 1)
    data = [steps, wrapper.total_train_losses]
    export_data = zip_longest(*data, fillvalue='')

    with open(os.path.join(data_dir, f'train_loss_csv_{wrapper.name}.csv'), 'w', encoding="ISO-8859-1",
              newline='') as f:
        write = csv.writer(f)
        write.writerow(['Step', 'Train_loss'])
        write.writerows(export_data)


def save_final_stats(wrapper, start_time):
    avg_return = sum(wrapper.total_avg_rewards) / len(wrapper.total_avg_rewards)

    last_5_percent_average_len = int(len(wrapper.total_avg_rewards) * 0.10)
    if last_5_percent_average_len == 0:
        last_5_percent_average_len = 1
    last_5_percent_train_loss_len = int(len(wrapper.total_train_losses) * 0.10)
    if last_5_percent_train_loss_len == 0:
        last_5_percent_train_loss_len = 1
    last_average = sum(wrapper.total_avg_rewards[-last_5_percent_average_len:]) / last_5_percent_average_len
    if len(wrapper.total_train_losses) != 0:
        avg_train_loss = sum(wrapper.total_train_losses) / len(wrapper.total_train_losses)
        last_train_loss = sum(
            wrapper.total_train_losses[-last_5_percent_train_loss_len:]) / last_5_percent_train_loss_len
    else:
        last_train_loss = 0
        avg_train_loss = 0

    total_time = t.time() - start_time

    with open(os.path.join(wrapper.wrapper_directory, f"final_stats_{wrapper.name}"), 'w') as f:
        f.write(f'Average Return: \n'
                f'{avg_return} \n'
                f'Last 10 Percent Average Return: \n'
                f'{last_average} \n'
                f'Average Train Loss: \n'
                f'{avg_train_loss} \n'
                f'Last 10 Percent Average Train Loss: \n'
                f'{last_train_loss} \n'
                f'Total Time for Training: \n'
                f'{convert_seconds(total_time)} \n'
                )


def convert_seconds(seconds):
    """Copied from https://www.geeksforgeeks.org/python-program-to-convert-seconds-into-hours-minutes-and-seconds/"""
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return "%d:%02d:%02d" % (hour, minutes, seconds)


def plot_avg_reward(wrapper, start_step, training_config):
    steps = range(start_step, start_step + training_config.num_iterations + 1, training_config.eval_interval)
    plt.plot(steps, wrapper.total_avg_rewards)
    plt.ylabel('Average Reward')
    plt.xlabel('Step')
    plt.axhline(y=np.nanmean(wrapper.total_avg_rewards))
    file_name = f'avg_reward.png'

    z = np.polyfit(steps, wrapper.total_avg_rewards, 1)
    p = np.poly1d(z)
    plt.plot(steps, p(steps), "r--")

    plt.savefig(os.path.join(wrapper.wrapper_directory, file_name))
    plt.clf()

def plot_avg_return(wrapper, start_step, training_config):
    steps = range(start_step, start_step + training_config.num_iterations + 1, training_config.eval_interval)
    plt.plot(steps, wrapper.total_avg_returns)
    plt.ylabel('Average Return')
    plt.xlabel('Step')
    plt.axhline(y=np.nanmean(wrapper.total_avg_returns))
    file_name = f'avg_return.png'

    z = np.polyfit(steps, wrapper.total_avg_returns, 1)
    p = np.poly1d(z)
    plt.plot(steps, p(steps), "r--")

    plt.savefig(os.path.join(wrapper.wrapper_directory, file_name))
    plt.clf()



def plot_replay_buffer_size(wrapper, start_step, training_config):
    steps = range(start_step, start_step + training_config.num_iterations + 1, 1)
    plt.plot(steps, wrapper.replay_buffer_sizes)
    plt.ylabel('Replay Buffer Size')
    plt.xlabel('Step')
    plt.ylim(bottom=0)
    file_name = f'replay_buffer_size.png'
    plt.savefig(os.path.join(wrapper.wrapper_directory, file_name))
    plt.clf()


def plot_train_loss(wrapper, start_step, training_config):
    steps = range(start_step + 1, start_step + training_config.num_iterations + 1, 1)
    plt.plot(steps, wrapper.total_train_losses)
    plt.ylabel('Train Loss')
    plt.xlabel('Step')
    plt.ylim(bottom=0)
    plt.axhline(y=np.nanmean(wrapper.total_train_losses))
    file_name = f'train_loss.png'
    plt.savefig(os.path.join(wrapper.wrapper_directory, file_name))
    plt.clf()


def plot_avg_reward_all_wrappers(directory, start_step, wrapper_list, training_config):
    steps = range(start_step, start_step + training_config.num_iterations + 1, training_config.eval_interval)
    for wrapper in wrapper_list:
        plt.plot(steps, wrapper.total_avg_rewards, label=wrapper.name)
        plt.ylabel('Average Reward')
        plt.xlabel('Step')
    file_name = f'avg_reward_all.png'
    plt.legend()
    plt.savefig(os.path.join(directory, file_name))
    plt.clf()

def plot_avg_return_all_wrappers(directory, start_step, wrapper_list, training_config):
    steps = range(start_step, start_step + training_config.num_iterations + 1, training_config.eval_interval)
    for wrapper in wrapper_list:
        plt.plot(steps, wrapper.total_avg_returns, label=wrapper.name)
        plt.ylabel('Average Return')
        plt.xlabel('Step')
    file_name = f'avg_return_all.png'
    plt.legend()
    plt.savefig(os.path.join(directory, file_name))
    plt.clf()


def plot_avg_accuracy_for_players(directory, start_step, training_config, players):
    steps = range(start_step, start_step + training_config.num_iterations + 1, training_config.eval_interval)
    for player in players:
        plt.plot(steps, player.avg_accuracies, label=player.name)
        plt.ylabel('Average Accuracy')
        plt.xlabel('Step')
        plt.ylim(top=105, bottom=-5)
        plt.axhline(y=np.nanmean(player.avg_accuracies))
        file_name = f'accuracy_{player.name}.png'
        plt.legend()
        z = np.polyfit(steps, player.avg_accuracies, 1)
        p = np.poly1d(z)
        plt.plot(steps, p(steps), "r--")
        plt.savefig(os.path.join(player.player_directory, file_name))
        plt.clf()

    for player in players:
        plt.plot(steps, player.avg_accuracies, label=player.name)
    plt.ylabel('Average Accuracy')
    plt.xlabel('Step')
    plt.ylim(top=105, bottom=-5)
    file_name = f'players/accuracy_all.png'
    plt.legend()

    plt.savefig(os.path.join(directory, file_name))
    plt.clf()


def plot_avg_score_for_players(directory, start_step, training_config, players):
    steps = range(start_step, start_step + training_config.num_iterations + 1, training_config.eval_interval)
    for player in players:
        plt.plot(steps, player.avg_points, label=player.name)
        plt.ylabel('Average Score')
        plt.xlabel('Step')
        plt.axhline(y=np.nanmean(player.avg_points))
        file_name = f'score_{player.name}.png'
        plt.legend()
        z = np.polyfit(steps, player.avg_points, 1)
        p = np.poly1d(z)
        plt.plot(steps, p(steps), "r--")

        plt.savefig(os.path.join(player.player_directory, file_name))
        plt.clf()

    for player in players:
        plt.plot(steps, player.avg_points, label=player.name)
    plt.ylabel('Average Score')
    plt.xlabel('Step')
    file_name = f'players/score_all.png'
    plt.legend()

    plt.savefig(os.path.join(directory, file_name))
    plt.clf()


def plot_avg_winrate_for_players(directory, start_step, training_config, players):
    steps = range(start_step, start_step + training_config.num_iterations + 1, training_config.eval_interval)
    for player in players:
        plt.plot(steps, player.winrates, label=player.name)
        plt.ylabel('Winrate')
        plt.xlabel('Step')
        plt.ylim(top=105, bottom=-5)
        plt.axhline(y=np.nanmean(player.winrates))
        file_name = f'winrate_{player.name}.png'
        plt.legend()
        z = np.polyfit(steps, player.winrates, 1)
        p = np.poly1d(z)
        plt.plot(steps, p(steps), "r--")

        plt.savefig(os.path.join(player.player_directory, file_name))
        plt.clf()

    for player in players:
        plt.plot(steps, player.winrates, label=player.name)
    plt.ylabel('Winrate')
    plt.xlabel('Step')
    plt.ylim(top=105, bottom=-5)
    file_name = f'players/winrate_all.png'
    plt.legend()

    plt.savefig(os.path.join(directory, file_name))
    plt.clf()


def save_config(directory, training_setup, bid_config, play_config, game_config, training_config):
    training_setup.save_config_to_file(directory)
    bid_config.save_config_to_file(directory, "Bid")
    play_config.save_config_to_file(directory, "Play")
    game_config.save_config_to_file(directory)
    training_config.save_config_to_file(directory)

def determine_save_file_path(training_setup):
    current_dir = os.getcwd()
    current_time = datetime.now().strftime("%H_%M")
    current_date = datetime.now().strftime("%d_%m")
    new_dir = os.path.join(current_dir, f'Tests/{current_date}/{current_time} {training_setup.name}')
    players_dir = os.path.join(new_dir, f'players')
    if not os.path.exists(players_dir):
        os.makedirs(players_dir)
    return new_dir

def save_player_data(start_step, training_config, players, game_config):
    for player in players:
        steps = range(start_step, start_step + training_config.num_iterations + 1, training_config.eval_interval)
        data_dir = os.path.join(player.player_directory, "data")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        data = [steps, player.avg_accuracies]
        export_data = zip_longest(*data, fillvalue='')

        with open(os.path.join(data_dir, f'avg_accuracy_csv_{player.name}.csv'), 'w', encoding="ISO-8859-1",
                  newline='') as f:
            write = csv.writer(f)
            write.writerow(['Step', 'Average Accuracy'])
            write.writerows(export_data)

        data = [steps, player.avg_points]
        export_data = zip_longest(*data, fillvalue='')

        with open(os.path.join(data_dir, f'avg_points_csv_{player.name}.csv'), 'w', encoding="ISO-8859-1",
                  newline='') as f:
            write = csv.writer(f)
            write.writerow(['Step', 'Avg Points'])
            write.writerows(export_data)

        plot_steps = [int(start_step) + 1, int(start_step) + int(training_config.num_iterations / 2),
                          int(start_step) + int(training_config.num_iterations)]


        for idx, plot_step in enumerate(plot_steps):
            with open(os.path.join(data_dir, f'guessed_tricks_{plot_step}_step_csv_{player.name}.csv'), 'w',
                      newline='') as f:
                write = csv.writer(f)
                write.writerows(player.history_guessed_tricks_sorted_after_round_number[idx])

            with open(os.path.join(data_dir, f'avg_accuracy_per_round_{plot_step}_step_csv_{player.name}.csv'), 'w',
                      newline='') as f:
                write = csv.writer(f)
                write.writerows(player.avg_accuracy_per_round[idx])

            with open(os.path.join(data_dir, f'boxplot_points_{plot_step}_step_csv_{player.name}.csv'), 'w',
                      newline='') as f:
                write = csv.writer(f)
                write.writerows(player.history_points_sorted_after_round_number[idx])

            with open(os.path.join(data_dir, f'avg_points_per_round_{plot_step}_step_csv_{player.name}.csv'), 'w',
                      newline='') as f:
                write = csv.writer(f)
                write.writerows(player.avg_points_per_round[idx])

            with open(os.path.join(data_dir, f'won_tricks_per_round_{plot_step}_step_csv_{player.name}.csv'), 'w',
                      newline='') as f:
                write = csv.writer(f)
                write.writerows(player.history_won_tricks_sorted_after_round_number[idx])



