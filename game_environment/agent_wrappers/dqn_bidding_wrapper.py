from tf_agents.agents import DqnAgent
from tf_agents.networks.q_network import QNetwork

from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.specs import tensor_spec
from tf_agents.networks import sequential

import tensorflow as tf

from game_environment.agent_wrappers.bidding_wrapper import BiddingWrapper
from game_environment.config_files.dqn_config import DQNConfig


class DQNBiddingWrapper(BiddingWrapper):

    def __init__(self, train_bidding_env, eval_bidding_env, dqn_config: DQNConfig, network):

        super().__init__()
        self.name = "DQN Bidding"

        self.set_bidding_env(train_bidding_env, eval_bidding_env)

        self.network = QNetwork(
            self.train_bidding_env.observation_spec(),
            self.train_bidding_env.action_spec(),
            fc_layer_params=dqn_config.fc_layer_params
        )

        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=dqn_config.learning_rate)

        self.train_step_counter = tf.Variable(0)

        self.agent = DqnAgent(
            self.train_bidding_env.time_step_spec(),
            self.train_bidding_env.action_spec(),
            q_network=self.network,
            optimizer=self.optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=self.train_step_counter)

        self.agent.initialize()

        self.reset_train_step_counter()

        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            batch_size=self.train_bidding_env.batch_size,
            max_length=dqn_config.replay_buffer_capacity)

        self.dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=3, sample_batch_size=dqn_config.batch_size,
            num_steps=dqn_config.n_step_update + 1).prefetch(3)

        self.dataset_iterator = iter(self.dataset)

        self.agent.train = common.function(self.agent.train)


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
            self.train_bidding_env.envs[0]._env.set_episode_ended(True)
            self.previous_bidding_time_step = self.current_bidding_time_step
            self.current_bidding_time_step = self.train_bidding_env.step(self.current_bidding_action_step.action)
            traj = trajectory.from_transition(self.previous_bidding_time_step, self.current_bidding_action_step,
                                              self.current_bidding_time_step)
            self.replay_buffer.add_batch(traj)
            # TODO: ADD ELSE CLAUSE
