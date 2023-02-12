import os.path

import tensorflow as tf
from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.networks import categorical_q_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

from game_environment.agent_wrappers.bidding_wrapper import BiddingWrapper
from game_environment.config_files.c51_config import C51Config
from game_environment.config_files.game_config import GameConfig


def observation_and_action_constraint_splitter(observation):
    return observation['observation'], observation['legal_moves']


class C51BiddingWrapper(BiddingWrapper):

    def __init__(self, player_name, train_bidding_env, eval_bidding_env, c51_config: C51Config, game_config: GameConfig, agent_mode, directory,
                 existing_model_mode="False", existing_model_path="", agent=None, network=None,
                 replay_buffer=None, dataset=None):

        super().__init__()
        self.global_step = None
        self.train_checkpointer = None
        self.dataset = None
        self.replay_buffer = None
        self.agent = None
        self.network = None
        self.epsilon_greedy = c51_config.epsilon_greedy
        self.number_of_rounds = game_config.number_of_total_rounds
        self.reward_function = c51_config.reward_function

        self.name = player_name + "_Bidding"

        self.directory = directory
        self.wrapper_directory = os.path.join(directory, f'wrappers/{self.name}')
        if not os.path.exists(self.wrapper_directory):
            os.makedirs(self.wrapper_directory)

        self.set_bidding_env(train_bidding_env, eval_bidding_env)
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=c51_config.learning_rate)

        self.allow_lr_decay = c51_config.allow_lr_decay
        self.lr_decay_rate = c51_config.lr_decay_rate
        self.lr_decay_threshold = c51_config.lr_decay_threshold
        self.min_lr = c51_config.min_lr

        self.allow_epsilon_decay = c51_config.allow_epsilon_decay
        self.epsilon_decay_rate = c51_config.epsilon_decay_rate
        self.epsilon_decay_threshold = c51_config.epsilon_decay_threshold
        self.min_epsilon = c51_config.min_epsilon

        self.replay_buffer_sizes = []
        self.history_learning_rates = [c51_config.learning_rate]
        self.history_epsilons = [c51_config.epsilon_greedy]

        self.train_step_counter = tf.Variable(0)

        if agent_mode == "SingleAgent" or network is None or replay_buffer is None or dataset is None:
            self.setup_single_agent(c51_config)
        else:
            self.setup_multi_agent(agent, network, replay_buffer, dataset)

        self.dataset_iterator = iter(self.dataset)

        self.agent.train = common.function(self.agent.train)

        if existing_model_mode is not None and existing_model_mode == "True" and agent is None:
            model_path = os.path.join(os.getcwd(), existing_model_path)
            self.create_check_pointer(model_path)
        elif agent is None:
            checkpoint_dir = os.path.join(self.wrapper_directory, f'checkpoint')
            self.create_check_pointer(checkpoint_dir)

        self.train_step_counter = self.agent.train_step_counter.numpy()

    def set_train_phase(self, train_phase: bool):
        self.train_phase = train_phase
        if train_phase == True:
            self.agent._epsilon_greedy = self.epsilon_greedy
        else:
            self.agent._epsilon_greedy = 0

    def create_check_pointer(self, checkpoint_dir):
        self.train_checkpointer = common.Checkpointer(ckpt_dir=checkpoint_dir,
                                                      max_to_keep=1,
                                                      agent=self.agent,
                                                      policy=self.agent.policy,
                                                      replay_buffer=self.replay_buffer,
                                                      global_step=self.train_step_counter)
        self.train_checkpointer.initialize_or_restore()

    def save_check_pointer(self):
        self.train_checkpointer.save(self.train_step_counter)

    def setup_single_agent(self, c51_config):
        self.network = categorical_q_network.CategoricalQNetwork(
            self.train_bidding_env.time_step_spec().observation['observation'],
            self.train_bidding_env.action_spec(),
            num_atoms=c51_config.num_atoms,
            fc_layer_params=c51_config.create_fc_layer_variable())

        if c51_config.min_q_value_auto:
            min_q_value = self.calculate_min_q()
        else:
            min_q_value = c51_config.min_q_value
        if c51_config.max_q_value_auto:
            max_q_value = self.calculate_max_q()
        else:
            max_q_value = c51_config.min_q_value

        self.agent = categorical_dqn_agent.CategoricalDqnAgent(
            self.train_bidding_env.time_step_spec(),
            self.train_bidding_env.action_spec(),
            categorical_q_network=self.network,
            observation_and_action_constraint_splitter=observation_and_action_constraint_splitter,
            optimizer=self.optimizer,
            min_q_value=min_q_value,
            max_q_value=max_q_value,
            n_step_update=c51_config.n_step_update,
            td_errors_loss_fn=common.element_wise_squared_loss,
            gamma=c51_config.gamma,
            train_step_counter=self.train_step_counter,
            epsilon_greedy=c51_config.epsilon_greedy)

        self.agent.initialize()

        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            batch_size=self.train_bidding_env.batch_size,
            max_length=c51_config.replay_buffer_capacity)

        self.dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=3, sample_batch_size=c51_config.batch_size,
            num_steps=c51_config.n_step_update + 1).prefetch(3)

    def calculate_min_q(self):
        if self.reward_function == "binary_reward":
            return 0
        if self.reward_function == "no_negative_reward":
            return 0
        if self.reward_function == "points_as_reward":
            return self.number_of_rounds * -10
        if self.reward_function == "points_as_reward_times_10":
            return (self.number_of_rounds * -10) * 10
        if self.reward_function == "normalize_reward":
            return -1.0

    def calculate_max_q(self):
        if self.reward_function == "binary_reward":
            return 1
        if self.reward_function == "no_negative_reward":
            return 1.0
        if self.reward_function == "points_as_reward":
            return 20 + self.number_of_rounds * 10
        if self.reward_function == "points_as_reward_times_10":
            return (20 + self.number_of_rounds * 10) * 10
        if self.reward_function == "normalize_reward":
            return 1.0

    def setup_multi_agent(self, agent, network, replay_buffer, dataset):
        self.network = network

        self.agent = agent

        self.replay_buffer = replay_buffer

        self.dataset = dataset

    def adjust_epsilon_greedy(self, current_step):
        if self.allow_epsilon_decay == "True":
            epsilon = self.epsilon_greedy
            print(f'Previous Epsilon: {epsilon}')
            self.epsilon_greedy = self.exp_decay(epsilon, current_step, self.epsilon_decay_threshold,
                                                self.epsilon_decay_rate, self.min_epsilon)
            print(f'New Epsilon: {self.epsilon_greedy}')
            self.history_epsilons.append(self.epsilon_greedy)
        else:
            self.history_epsilons.append(self.epsilon_greedy)
        return

    def adjust_learning_rate(self, current_step):
        if self.allow_lr_decay == "True":
            learning_rate = self.optimizer._lr
            print(f'Previous Learning Rate: {learning_rate}')
            self.optimizer._lr = self.exp_decay(learning_rate, current_step, self.lr_decay_threshold,
                                                self.lr_decay_rate, self.min_lr)
            print(f'New Learning Rate: {self.optimizer._lr}')
            self.history_learning_rates.append(self.optimizer._lr)
        else:
            self.history_learning_rates.append(self.optimizer._lr)

    @staticmethod
    def exp_decay(variable, step, decay_threshold, decay_rate, min_value):
        if step < decay_threshold or variable <= min_value:
            return variable
        else:
            new_variable = variable * tf.math.exp(decay_rate)
            if new_variable <= min_value:
                return min_value
            else:
                return new_variable

    def train_agent(self):
        """Sample a batch of data from the buffer and update the agent's network."""
        experience, unused_info = next(self.dataset_iterator)
        train_loss = self.agent.train(experience)
        self.total_train_losses.append(train_loss.loss)
        return train_loss.loss

    def bidding_train_phase(self):
        if self.current_bidding_action_step is not None:
            self.train_bidding_env.step(self.current_bidding_action_step.action)
        self.previous_bidding_time_step = self.current_bidding_time_step
        self.current_bidding_time_step = self.train_bidding_env.current_time_step()
        if self.previous_bidding_time_step is not None:
            traj = trajectory.from_transition(self.previous_bidding_time_step, self.current_bidding_action_step,
                                              self.current_bidding_time_step)
            self.replay_buffer.add_batch(traj)

        self.current_bidding_action_step = self.policy.action(self.current_bidding_time_step)
        return self.current_bidding_action_step.action.numpy()[0]

    def wrap_up_game(self):
        if self.train_phase:
            self.train_bidding_env.envs[0].set_episode_ended(True)
            self.previous_bidding_time_step = self.current_bidding_time_step
            self.current_bidding_time_step = self.train_bidding_env.step(self.current_bidding_action_step.action)
            traj = trajectory.from_transition(self.previous_bidding_time_step, self.current_bidding_action_step,
                                              self.current_bidding_time_step)
            self.replay_buffer.add_batch(traj)
        else:
            self.eval_bidding_env.envs[0].set_episode_ended(True)
            self.eval_bidding_env.step(self.current_bidding_action_step.action)
            current_time_step = self.eval_bidding_env.current_time_step()
            self.rewards.append(current_time_step.reward)

    def print_current_replay_buffer_size(self):
        self.replay_buffer_sizes.append(int(self.replay_buffer.num_frames()))
        print(f"{self.name} current Replay Buffer Size: {self.replay_buffer_sizes[-1]}")

    def save_return_of_episode(self):
        self.total_returns.append(float(sum(self.rewards)))
