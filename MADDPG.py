import tensorflow as tf
from keras.optimizers import Adam
import numpy as np
from Utils import ActorNetwork, CriticNetwork, ReplayBuffer, merge_actions
from itertools import chain
from copy import deepcopy
from constants import NUM_AGENTS, STATE_DIM, ACTION_DIM


class MADDPG(object):
  # state_dim = 4*N
  # action_dim = 3*N
  def __init__(self, env, n_agents, state_dim=STATE_DIM, action_dim=ACTION_DIM, mem_capacity=250000,
               roll_out_n_steps=10, batch_size=100, episodes=2000, timeslots=200, epsilon_start=0.9,
               epsilon_end=0.01, epsilon_decay=200, max_steps=100):
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
    self.max_steps = max_steps

    self.epsilon_end = epsilon_end
    self.epsilon_start = epsilon_start
    self.epsilon_decay = epsilon_decay

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


  def _take_one_step(self):
    state = self.env.get_state()
    # exploration_action finds the action for all users using their local networks
    action = self.exploration_action()
    next_state, reward = self.env.step(action)
    self.n_steps += 1
    self.memory.push(state, action, reward, next_state)


  # should be used after collecting some experience
  def train(self):
    for episode in range(self.episodes):
      print("Episode {:>5}/{}".format(episode, self.episodes))
      self.env.reset()

      for timeslot in range(self.timeslots):
        print("\tTimeslot {:>4}/{}".format(timeslot, self.timeslots))

        state = self.env.get_state()
        action = self.exploration_action()
        next_state, reward = self.env.step(action)
        self.n_steps += 1

        exp_tuple = (state, action, reward, next_state)
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
    # TODO: change random noise to Ornstein-Uhlenbeck process
    actions = self.action()
    epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                              np.exp(-1. * self.n_steps / self.epsilon_decay)
    noise = np.random.randn(self.action_dim * self.n_agents) * epsilon
    # noise should be added to each action
    assert(np.shape(actions)==np.shape(noise))

    return actions + noise