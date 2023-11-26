from constants import NUM_USERS, STATE_DIM, ACTION_DIM, BETA
import numpy as np
import tensorflow as tf
from keras import losses, optimizers
from Utils import ReplayBuffer
from Networks import Actor, Critic
from copy import deepcopy


class MADDPG():
  def __init__(self, env):
    self.n_users = NUM_USERS
    self.memory_capacity = 250000
    self.rollout_steps = 10
    self.env = env
    self.n_steps = 0
    self.batch_size = 100
    self.gamma = 0.99
    self.tau = 0.005
    self.epsilon_start = 0.9
    self.epsilon_end = 0.01
    self.epsilon_decay = 200

    self.actors = [Actor(STATE_DIM, ACTION_DIM) for _ in range(self.n_users)]
    self.critics = [Critic(self.n_users, STATE_DIM, ACTION_DIM) for _ in range(self.n_users)]
    self.target_actors = deepcopy(self.actors)
    self.target_critics = deepcopy(self.critics)
    self.local_actors = [Actor(STATE_DIM, ACTION_DIM) for _ in range(self.n_users)]

    self.actor_optimizer = [optimizers.Adam(learning_rate=0.001) for _ in range(self.n_users)]
    self.critic_optimizer = [optimizers.Adam(learning_rate=0.002) for _ in range(self.n_users)]

    self.replay_buffer = ReplayBuffer(self.memory_capacity)


  def rollout(self):
    for _ in range(self.rollout_steps):
      self.n_steps += 1
      
      state = self.env.get_state()
      action = self.get_action(state)
      next_state, reward = self.env.step(action)
      self.store_experience(state, action, reward, next_state)
    print("Rollout complete")


  def _get_epsilon(self):
    epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                              np.exp(-1. * self.n_steps / self.epsilon_decay)
    return epsilon


  def get_action(self, state, exploration=True):
    action = np.array([])
    for user_id in range(NUM_USERS):
      user_action = self.local_actors[user_id].predict(state.reshape(1,-1), verbose=0)[0]
      action = np.concatenate((action, user_action))

    if exploration:
      epsilon = self._get_epsilon()
      noise = np.random.normal(scale=0.1, size=ACTION_DIM * NUM_USERS) * epsilon
      action += noise

    return action


  def store_experience(self, state, action, reward, next_state):
    self.replay_buffer.push(state, action, reward, next_state)


  def get_actor_action(self, state):
    action = np.array([])
    for user in range(self.n_users):
      user_action = self.actors[user].predict(state.reshape(1,-1), verbose=0)[0]
      action = np.concatenate((action, user_action))
    return action
  

  def update_target_networks(self):
    for user in range(self.n_users):
      actor_weights = self.actors[user].get_weights()
      target_actor_weights = self.target_actors[user].get_weights()
      critic_weights = self.critics[user].get_weights()
      target_critic_weights = self.target_critics[user].get_weights()

      for i in range(len(actor_weights)):
        target_actor_weights[i] = self.tau * actor_weights[i] + (1 - self.tau) * target_actor_weights[i]
      for i in range(len(critic_weights)):
        target_critic_weights[i] = self.tau * critic_weights[i] + (1 - self.tau) * target_critic_weights[i]
      
      self.target_actors[user].set_weights(target_actor_weights)
      self.target_critics[user].set_weights(target_critic_weights)


  def update_local_networks(self):
    for user in range(self.n_users):
      target_actor_weights = self.target_actors[user].get_weights()
      self.local_actors[user].set_weights(target_actor_weights)


  def train(self):
    self.n_steps += 1

    mini_batch = self.replay_buffer.sample(self.batch_size)
    states = np.vstack(mini_batch.states)
    actions = np.vstack(mini_batch.actions)
    rewards = np.vstack(mini_batch.rewards)
    next_states = np.vstack(mini_batch.next_states)
    target_actions = []
    # updated_actor_actions = []

    target_actions = [self.target_actors[x].predict(next_states) for x in range(self.n_users)]
    target_actions = np.concatenate(target_actions, axis=1)
    # for state in states:
    #   updated_actor_actions.append(self.get_actor_action(state))
    # target_actions = np.vstack(target_actions)
    # updated_actor_actions = np.vstack(updated_actor_actions)

    for user in range(self.n_users):
      with tf.GradientTape() as tape:
        current_q = self.critics[user](tf.concat([states, actions], axis=-1))
        target_q = self.target_critics[user](tf.concat([next_states, target_actions], axis=-1))
        y_val = rewards + self.gamma * target_q
        critic_loss = losses.mean_squared_error(y_val, current_q)
      critic_gradient = tape.gradient(critic_loss, self.critics[user].trainable_variables)
      self.critic_optimizer[user].apply_gradients(zip(critic_gradient, self.critics[user].trainable_variables))
    
      with tf.GradientTape() as tape:
        user_actions = self.actors[user](states)
        new_actions = deepcopy(actions)
        new_actions[:, user:user+ACTION_DIM] = user_actions
        new_actions = tf.Variable(new_actions)
        q_val = self.critics[user](tf.concat([states, new_actions], axis=-1))
      actor_gradient = tape.gradient(q_val, new_actions)

      with tf.GradientTape() as tape:
        policy = self.actors[user](states)
      policy_gradient = tape.gradient(policy, self.actors[user].trainable_variables)
      1/0
      self.actor_optimizer[user].apply_gradients(zip(actor_gradient, self.actors[user].trainable_variables))


    self.update_target_networks(self)
    if self.n_steps % BETA == 0:
      self.update_local_networks()