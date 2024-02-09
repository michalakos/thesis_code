from models import QNetwork
from torch.optim import Adam
import torch
import torch.nn as nn
import numpy as np
from constants import *
from environment import Environment
from memory import ReplayMemory


class DQNAgent:
  def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, epsilon_start=1.0, epsilon_decay=0.995, epsilon_end=0.01):
    self.state_size = state_size
    self.action_size = action_size
    self.epsilon_decay = epsilon_decay
    self.epsilon_start = epsilon_start
    self.epsilon_end = epsilon_end
    self.gamma = gamma
    self.steps = 0
    self.num_users = NUM_USERS
    self.capacity = CAPACITY
    self.batch_size = BATCH_SIZE

    # Q-networks
    self.policy_nets = [QNetwork(state_size, action_size) for _ in self.num_users]
    self.target_nets = [QNetwork(state_size, action_size) for _ in self.num_users]
    for user_i in range(self.num_users):
      self.target_nets[user_i].load_state_dict(self.policy_nets[user_i].state_dict())

    # Optimizer
    self.optimizers = [Adam(self.policy_nets[i].parameters(), lr=learning_rate, amsgrad=True) for i in range(self.num_users)]

    self.memory = ReplayMemory(self.capacity)


  def select_action(self, state):
    eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
      np.exp(-1. * self.steps / self.epsilon_decay)
    self.steps += 1

    if np.random.rand() <= eps_threshold:
      action = np.random.choice(self.action_size)
    
    else:
      for user in range(self.num_users):
        with torch.no_grad():
          self.policy_nets[user](state).max(1).indices.view(1, 1)
    

  def update_model(self, state, action, reward, next_state):
    state = torch.tensor(state, dtype=torch.float32)
    next_state = torch.tensor(next_state, dtype=torch.float32)

    q_values = self.q_network(state)
    next_q_values = self.target_q_network(next_state)

    target = q_values.clone()
    target[action] = reward + self.gamma * torch.max(next_q_values).item()

    loss = nn.MSELoss()(q_values, target)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    
  def update_target_network(self):
    self.target_q_network.load_state_dict(self.q_network.state_dict())


num_agents = NUM_USERS
state_size = STATE_DIM
action_size = ACTION_DIM
num_episodes = EPISODES
num_times = TIMESLOTS

agent = DQNAgent(state_size=state_size, action_size=action_size)
env = Environment()

for episode in range(1, num_episodes+1):
  obs = env.reset()
  obs = env.get_state()
  obs = np.stack(obs)
  if isinstance(obs, np.ndarray):
    obs = torch.from_numpy(obs).float()
  total_reward = 0.0
  for t in range(num_times):
    obs = obs.type(torch.FloatTensor)
    action = agent.select_action(obs).data.cpu()

  for agent in agents:
    state = env.reset()