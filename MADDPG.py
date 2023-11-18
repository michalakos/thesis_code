import tensorflow as tf
from keras.optimizers import Adam
import numpy as np
from Agents import ActorNetwork, CriticNetwork, ReplayBuffer
from itertools import chain
from copy import deepcopy


class MADDPG(object):
  # state_dim = 4*N
  # action_dim = 3*N
  def __init__(self, env, n_agents, state_dim=16, action_dim=12, mem_capacity=250000,
               roll_out_n_steps=10, batch_size=100,
               episodes=2000, timeslots=200):
    self.env = env
    self.n_agents = n_agents
    self.state_dim = state_dim
    self.action_dim = action_dim
    self.mem_capacity = mem_capacity
    self.roll_out_n_steps = roll_out_n_steps
    self.batch_size = batch_size
    self.episodes = episodes
    self.timeslots = timeslots
    self.n_steps = 0

    self.epsilon_end = 0.01
    self.epsilon_start = 0.9
    self.epsilon_decay = 200

    # parameter sharing has only one target actor and one target critic networks
    # and a local actor network at each user
    self.local_actors = [ActorNetwork() for id in range(self.n_agents)]
    self.actor = ActorNetwork()
    self.actor_target = deepcopy(self.actor)
    self.critic = CriticNetwork()
    self.critic_target = deepcopy(self.critic)

    self.memory = ReplayBuffer(self.mem_capacity)

    self.actor_optimizer = Adam()
    self.critic_optimizer = Adam()


  def train(self):
    for episode in range(self.episodes):
      print("Episode {:>5}/{}".format(episode, self.episodes))
      self.env.reset()

      for timeslot in range(self.timeslots):
        print("\tTimeslot {:>4}/{}".format(timeslot, self.timeslots))

        actions = []
        states = []
        for user_id in range(self.n_agents):
          state_k = self.env.get_state_k(user_id)
          local_actor_k = self.local_actors[user_id]
          action_k = local_actor_k.select_action(state_k)
          
          actions.append(action_k)
          states.append(state_k)

        next_state, reward = self.env.step(actions)
        # from ((p1_1, p1_2, S1), (p2_1, p2_2, S2), ...)
        # to (p1_1, p2_1, ..., p1_2, p2_2, ..., S1, S2, ...)
        actions = tuple(chain(*zip(*actions)))
        states = tuple(chain(*zip(*states)))

        exp_tuple = (states, actions, reward, next_state)
        self.memory.push(*exp_tuple)

        samples = self.memory.sample(self.batch_size)
        # update critic
        # update actor
        # update target networks
        # update local actor every beta slots


  def action(self):
    actions = []
    for user_id in range(self.n_agents):
      state_k = self.env.get_state_k()
      action_var = self.local_actors[user_id](tf.convert_to_tensor(state_k))
      action_k = action_var.data.numpy()[0]
      actions.append(action_k)

    actions = tuple(chain(*zip(*actions)))
    return actions
  

  def exploration_action(self):
    actions = self.action()
    epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                              np.exp(-1. * self.n_steps / self.epsilon_decay)
    noise = np.random.randn(self.action_dim) * epsilon

    return actions + noise