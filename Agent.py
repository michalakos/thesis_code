import tensorflow as tf
from keras import optimizers
import numpy as np
from itertools import chain
import random
from Networks import build_actor, build_critic
from Utils import ReplayBuffer
from constants import STATE_DIM, ACTION_DIM, NUM_USERS, BETA

CRITIC_STATE_DIM = (STATE_DIM + 1) * NUM_USERS
CRITIC_ACTION_DIM = ACTION_DIM * NUM_USERS


class DDPGAgent:
  def __init__(self):
    self.batch_size = 100
    self.memory_capacity = 250000
    self.episodes = 2000
    self.timeslots = 200
    self.n_steps = 0
    self.roll_out_steps = 10
    self.epsilon_start = 0.9
    self.epsilon_end = 0.01
    self.epsilon_decay = 200
    self.gamma = 0.99
    self.tau = 0.005

    self.actor = build_actor(state_dim=STATE_DIM, action_dim=ACTION_DIM)
    self.critic = build_critic(state_dim=CRITIC_STATE_DIM, action_dim=CRITIC_ACTION_DIM)
    self.target_actor = build_actor(state_dim=STATE_DIM, action_dim=ACTION_DIM)
    self.target_critic = build_critic(state_dim=CRITIC_STATE_DIM, action_dim=CRITIC_ACTION_DIM)

    self.target_actor.set_weights(self.actor.get_weights())
    self.target_critic.set_weights(self.critic.get_weights())

    self.actor_optimizer = optimizers.Adam(learning_rate=0.001)
    self.critic_optimizer = optimizers.Adam(learning_rate=0.002)

    self.local_actors = []

    for user_id in range(NUM_USERS):
      self.local_actors.append(build_actor(state_dim=STATE_DIM, action_dim=ACTION_DIM))

    self.memory = ReplayBuffer(self.memory_capacity)


  def update_target_networks(self):
    actor_weights = self.actor.get_weights()
    target_actor_weights = self.target_actor.get_weights()
    critic_weights = self.critic.get_weights()
    target_critic_weights = self.target_critic.get_weights()

    for i in range(len(actor_weights)):
      target_actor_weights[i] = self.tau * actor_weights[i] + (1 - self.tau) * target_actor_weights[i]

    for i in range(len(critic_weights)):
            target_critic_weights[i] = self.tau * critic_weights[i] + (1 - self.tau) * target_critic_weights[i]

    self.target_actor.set_weights(target_actor_weights)
    self.target_critic.set_weights(target_critic_weights)


  def update_local_networks(self):
    target_actor_weights = self.target_actor.get_weights()
    for user_id in range(NUM_USERS):
      self.local_actors[user_id].set_weights(target_actor_weights)



  def _get_user_state(self, state, user_id):
     user_state = (state[user_id], state[NUM_USERS+user_id], state[2*NUM_USERS+user_id])
     return np.array(user_state)
  

  def _get_epsilon(self):
    epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                              np.exp(-1. * self.n_steps / self.epsilon_decay)
    return epsilon


  def get_action(self, state):
    user_actions = []
    for user_id in range(NUM_USERS):
      user_state = self._get_user_state(state, user_id)
      user_action = self.local_actors[user_id].predict(user_state.reshape(1,-1))[0]
      user_actions.append(user_action)

    action = tuple(chain(*zip(*user_actions)))
    epsilon = self._get_epsilon()
    noise = np.random.normal(scale=0.1, size=ACTION_DIM * NUM_USERS) * epsilon
    return action + noise
  

  def store_experience(self, state, action, reward, next_state):
    self.memory.push(state, action, reward, next_state)
  

  def train(self):
    self.n_steps += 1

    user_id = random.randint(0, NUM_USERS-1)
    mini_batch = self.memory.sample(self.batch_size)
    user_states = [self._get_user_state(state, user_id) for state in mini_batch.states]
    user_states = np.vstack(user_states)
    user_next_states = [self._get_user_state(next_state, user_id) for next_state in mini_batch.next_states]
    user_next_states = np.vstack(user_next_states)
    states = np.vstack(mini_batch.states)
    actions = np.vstack(mini_batch.actions)
    rewards = np.vstack(mini_batch.rewards)
    next_states = np.vstack(mini_batch.next_states)

    with tf.GradientTape() as tape:
      # TODO: calculate target action for all users
      target_actions = self.target_actor(user_next_states)
      target_q_values = self.target_critic(tf.concat([next_states, target_actions], axis=1))
      target_values = rewards + self.gamma * target_q_values
      predicted_values = self.critic(tf.concat([states, actions], axis=1))
      critic_loss = tf.reduce_mean(tf.square(predicted_values - target_values))

    critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
    self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))

    with tf.GradientTape() as tape:
      actions = self.actor(user_states)
      q_values = self.critic(tf.concat([states, actions], axis=1))
      actor_loss = -tf.reduce_mean(q_values)

    actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
    self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))

    self.update_target_networks()
    if self.n_steps % BETA == 0:
      self.update_local_networks()
      