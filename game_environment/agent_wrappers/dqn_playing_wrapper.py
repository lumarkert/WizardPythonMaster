import tensorflow as tf
from tf_agents.agents import DqnAgent
from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.networks import sequential
from tf_agents.networks.q_network import QNetwork
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

from game_environment.agent_wrappers.dqn_bidding_wrapper import DQNConfig
from game_environment.agent_wrappers.playing_wrapper import PlayingWrapper
from game_environment.config_files.c51_config import C51Config


def observation_and_action_constraint_splitter(observation):
    return observation['observation'], observation['legal_moves']


class DQNPlayingWrapper(PlayingWrapper):

    def __init__(self, train_playing_env, eval_playing_env, dqn_config: DQNConfig, network):

        super().__init__()
        self.name = "DQN Playing"

        self.set_playing_env(train_playing_env, eval_playing_env)

        self.network = QNetwork(
            self.train_playing_env.time_step_spec().observation['observation'],
            self.train_playing_env.action_spec(),
            fc_layer_params=dqn_config.fc_layer_params
        )

        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=dqn_config.learning_rate)

        self.train_step_counter = tf.Variable(0)

        self.agent = DqnAgent(
            self.train_playing_env.time_step_spec(),
            self.train_playing_env.action_spec(),
            q_network=self.network,
            optimizer=self.optimizer,
            observation_and_action_constraint_splitter=observation_and_action_constraint_splitter,
            gamma=dqn_config.gamma,
            td_errors_loss_fn=common.element_wise_squared_loss,
            epsilon_greedy=dqn_config.epsilon_greedy,
            train_step_counter=self.train_step_counter)

        self.agent.initialize()

        self.reset_train_step_counter()

        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            batch_size=self.train_playing_env.batch_size,
            max_length=dqn_config.replay_buffer_capacity)

        self.dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=3, sample_batch_size=dqn_config.batch_size,
            num_steps=dqn_config.n_step_update + 1).prefetch(3)

        self.agent.train = common.function(self.agent.train)

        self.dataset_iterator = iter(self.dataset)

        self.agent.train = common.function(self.agent.train)


    def train_agent(self):
        experience, unused_info = next(self.dataset_iterator)
        train_loss = self.agent.train(experience)
        self.total_train_losses.append(train_loss.loss)
        return train_loss.loss

    def playing_train_phase(self):
        if self.current_playing_action_step is not None:
            self.train_playing_env.step(self.current_playing_action_step.action)
        self.previous_playing_time_step = self.current_playing_time_step
        self.current_playing_time_step = self.train_playing_env.current_time_step()
        if self.previous_playing_time_step is not None:
            traj = trajectory.from_transition(self.previous_playing_time_step, self.current_playing_action_step,
                                              self.current_playing_time_step)

            self.replay_buffer.add_batch(traj)

        self.current_playing_action_step = self.policy.action(self.current_playing_time_step)
        return self.current_playing_action_step.action.numpy()[0]

    def wrap_up_round(self, current_trick, playable_cards):
        self.set_current_trick(current_trick)
        self.set_playable_cards(playable_cards)
        if self.train_phase:
            self.train_playing_env.pyenv.envs[0].set_episode_ended(True)
            self.previous_playing_time_step = self.current_playing_time_step
            self.current_playing_time_step = self.train_playing_env.step(self.current_playing_action_step.action)
            traj = trajectory.from_transition(self.previous_playing_time_step, self.current_playing_action_step,
                                              self.current_playing_time_step)
            self.replay_buffer.add_batch(traj)
        else:
            self.eval_playing_env.pyenv.envs[0].set_episode_ended(True)
            self.eval_playing_env.step(self.current_playing_action_step.action)
            current_time_step = self.eval_playing_env.current_time_step()
            self.rewards.append(current_time_step.reward)

    def create_checkpoint(self, directory):
        common.Checkpointer(
            ckpt_dir=directory,
            max_to_keep=1,
            agent=self.agent,
            policy=self.agent.policy,
            replay_buffer=self.replay_buffer
            #global_step=global_step
        )