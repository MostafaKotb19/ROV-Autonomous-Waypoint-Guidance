import numpy as np
import tensorflow as tf
from collections import deque
import datetime
import os

class PPO_agent:
    def __init__(self, num_actions, observation_space_size):
        """
        Initialize the PPOAgent.

        Initializes the agent with neural networks, optimizers, memory buffer, and hyperparameters.

        Parameters:
        - num_actions: Number of possible actions in the environment.
        - observation_space_size: Size of the observation space.
        """
        self.num_actions = num_actions
        self.observation_space_size = observation_space_size
        self.policy = self.build_policy_network()
        self.old_policy = self.build_policy_network()
        self.policy_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.value_network = self.build_value_network()
        self.value_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.epsilon = 0.2
        self.batch_size = 32
        self.log_file = None

    def create_log_file(self):
        """Create a log file with a timestamp for recording training progress."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"loss_log_{timestamp}.txt"
        self.log_file = open(log_filename, "a")

    def close_log_file(self):
        """Close the log file."""
        if self.log_file:
            self.log_file.close()

    def log_episode_reward(self, episode_num, total_reward):
        """
        Log episode reward to the file.

        Parameters:
        - episode_num: Episode number.
        - total_reward: Total reward obtained in the episode.
        """
        if self.log_file is None:
            self.create_log_file()

        self.log_file.write(f"Episode {episode_num}, Total Reward: {total_reward}\n")
        self.log_file.flush()

    def build_policy_network(self):
        """
        Build the policy neural network model.

        Returns:
        - model: Policy neural network model.
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.num_actions, activation='linear')
        ])
        return model

    def build_value_network(self):
        """
        Build the value neural network model.

        Returns:
        - model: Value neural network model.
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        return model

    def select_action(self, state):
        """
        Select an action based on the current state.

        Parameters:
        - state: Current state.

        Returns:
        - action: Selected action.
        """
        state = np.array([state])
        action = self.policy(state)[0].numpy()
        action = np.concatenate([np.repeat(action[4], 4, axis=0), action[:4]])
        action = 50 * action
        return action

    def remember(self, state, action, reward, next_state, done):
        """
        Store an experience tuple in the memory buffer.

        Parameters:
        - state: Current state.
        - action: Taken action.
        - reward: Received reward.
        - next_state: Next state.
        - done: Whether the episode is done or not.
        """
        self.memory.append((state, action, reward, next_state, done))

    def discounted_rewards(self, rewards):
        """
        Calculate discounted rewards for a sequence of rewards.

        Parameters:
        - rewards: Sequence of rewards.

        Returns:
        - discounted: Discounted rewards.
        """
        discounted = np.zeros_like(rewards, dtype=np.float32)
        running_add = 0
        for t in reversed(range(len(rewards))):
            running_add = running_add * self.gamma + rewards[t]
            discounted[t] = running_add
        return discounted

    def compute_advantages(self, rewards, values, dones):
        """
        Compute advantages based on rewards and predicted values.

        Parameters:
        - rewards: Sequence of rewards.
        - values: Predicted values.
        - dones: Whether the episode is done or not.

        Returns:
        - advantages: Computed advantages.
        """
        discounted_rewards = self.discounted_rewards(rewards)
        advantages = discounted_rewards - values
        return advantages

    def update_policy(self, states, actions, advantages, old_probs, episode_num):
        """
        Update the policy neural network based on PPO loss.

        Parameters:
        - states: States from the memory buffer.
        - actions: Actions from the memory buffer.
        - advantages: Computed advantages.
        - old_probs: Old probabilities from the memory buffer.
        - episode_num: Episode number.
        """
        with tf.GradientTape() as tape:
            new_probs = self.policy(states)
            action_masks = tf.one_hot(actions, self.num_actions)

            ratios = new_probs / (old_probs + 1e-8)
            clipped_ratios = tf.clip_by_value(ratios, 1 - self.epsilon, 1 + self.epsilon)

            # Ensure that advantages have shape [batch_size, 1]
            advantages = tf.expand_dims(advantages, axis=-1)

            surrogate1 = ratios * advantages
            surrogate2 = clipped_ratios * advantages
            policy_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))

        gradients = tape.gradient(policy_loss, self.policy.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(gradients, self.policy.trainable_variables))

        if self.log_file is None:
            self.create_log_file()
        self.log_file.write(f"Episode {episode_num}, Policy Loss: {policy_loss.numpy()}\n")
        self.log_file.flush()

    def update_value_network(self, states, discounted_rewards, episode_num):
        """
        Update the value network based on mean squared error loss.

        Parameters:
        - states: States from the memory buffer.
        - discounted_rewards: Discounted rewards.
        - episode_num: Episode number.
        """
        with tf.GradientTape() as tape:
            values = self.value_network(states)
            value_loss = tf.reduce_mean(tf.square(discounted_rewards - values))

        gradients = tape.gradient(value_loss, self.value_network.trainable_variables)
        self.value_optimizer.apply_gradients(zip(gradients, self.value_network.trainable_variables))

        if self.log_file is None:
            self.create_log_file()
        self.log_file.write(f"Episode {episode_num}, Value Loss: {value_loss.numpy()}\n")
        self.log_file.flush()

    def save_model(self, episode_num):
        """
        Save the policy and value network models.

        Parameters:
        - episode_num: Episode number.
        """
        model_dir = "model_checkpoints"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        policy_model_filename = f"{model_dir}/policy_model_episode_{episode_num}.h5"
        value_model_filename = f"{model_dir}/value_model_episode_{episode_num}.h5"

        self.policy.save(policy_model_filename)
        self.value_network.save(value_model_filename)

    def load_model(self, episode_num):
        """
        Load saved policy and value network models.

        Parameters:
        - episode_num: Episode number.
        """
        if episode_num == 0:
            return
        model_dir = "model_checkpoints"

        policy_model_filename = f"{model_dir}/policy_model_episode_{episode_num}.h5"
        value_model_filename = f"{model_dir}/value_model_episode_{episode_num}.h5"

        if os.path.exists(policy_model_filename) and os.path.exists(value_model_filename):
            self.policy = tf.keras.models.load_model(policy_model_filename)
            self.value_network = tf.keras.models.load_model(value_model_filename)
            print(f"Models loaded from episode {episode_num}")
        else:
            print("No saved models found.")