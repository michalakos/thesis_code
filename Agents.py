# code inspired by https://github.com/ChenglongChen/pytorch-DRL
K = 4

import tensorflow as tf
from tensorflow.python.keras.layers import Dense, InputLayer, Concatenate
from tensorflow.python.keras import Model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.initializers import RandomNormal

import random
from collections import namedtuple

# channel_gain_BS, channel_gain_EVE, task_size, decoding_order
state_size = 4
# total_power, first_message_power_ratio, task_size_ratio
action_size = 3


# the actor network receives the state as an input
# has two hidden layers of size 400 and 300 respectively with ReLU activation
# has sigmoid output of size equal to the number of actions
class ActorNetwork(Model):
  def __init__(self):
    self.N_users = K
    initializer = RandomNormal()

    super().__init__()
    self.input_layer = InputLayer(state_size)
    self.hidden_1 = Dense(400, activation='relu',
                          kernel_initializer=initializer)
    self.hidden_2 = Dense(300, activation='relu',
                          kernel_initializer=initializer)
    self.output = Dense(action_size, activation='sigmoid',
                        kernel_initializer=initializer)


  def call(self, state):
    x = self.input_layer(state)
    x = self.hidden_1(x)
    x = self.hidden_2(x)
    return self.output(x)


  def take_action(self, state): #TODO
    action = 0
    return action


# the critic network receives the global state as an input and
# the action taken as an input in the second layer
# has two hidden layers of size 400 and 300 respectively with ReLU activation
# has single output returning loss function
class CriticNetwork(Model):
  def __init__(self):
    self.N_users = K
    initializer = RandomNormal()

    super().__init__()
    self.input_layer = InputLayer(state_size * self.N_users)
    self.hidden_1 = Dense(400, activation='relu',
                          kernel_initializer=initializer)
    self.action_layer = InputLayer(action_size * self.N_users)
    self.concat = Concatenate()
    self.hidden_2 = Dense(300, activation='relu',
                          kernel_initializer=initializer)
    self.output = Dense(1, kernel_initializer=initializer)

  def call(self, state, actions):
    x = self.input_layer(state)
    x = self.hidden_1(x)
    acts = self.action_layer(actions)
    x = self.concat([x, acts])
    x = self.hidden_2(x)
    return self.output(x)


# replay buffer holds the total experience from all users for each timestep
# basically a FIFO queue
Experience = namedtuple(
    "Experience", ("states", "actions", "rewards", "next_states"))


class ReplayBuffer(object):
  def __init__(self, capacity):
    self.capacity = capacity
    self.memory = []
    self.position = 0

  def _push_one(self, state, action, reward, next_state=None):
    if len(self.memory) < self.capacity:
      self.memory.append(None)
    self.memory[self.position] = Experience(state, action, reward, next_state)
    self.position = (self.position + 1) % self.capacity

  def push(self, states, actions, rewards, next_states=None):
    if isinstance(states, list):
      if next_states is not None and len(next_states) > 0:
        for s,a,r,n_s in zip(states, actions, rewards, next_states):
          self._push_one(s, a, r, n_s)
      else:
        for s,a,r in zip(states, actions, rewards):
          self._push_one(s, a, r)
    else:
      self._push_one(states, actions, rewards, next_states)

  def sample(self, batch_size):
    if batch_size > len(self.memory):
      batch_size = len(self.memory)
    transitions = random.sample(self.memory, batch_size)
    batch = Experience(*zip(*transitions))
    return batch

  def __len__(self):
    return len(self.memory)