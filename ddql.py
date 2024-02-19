from models import DQN
from torch.optim import Adam
from torch import FloatTensor
import torch
import torch.nn as nn
import numpy as np
from constants import *
from environment import Environment
from memory import ReplayMemory, Experience
import random
from datetime import datetime
import os
from model_utils import save_model


device = torch.device("cpu")


class DDQN:
  def __init__(self, env, learning_rate=1e-4, gamma=0.99, tau=0.005,
               epsilon_start=0.9, epsilon_decay=500000, epsilon_end=0.05):
    self.capacity = 100000
    self.batch_size = 1024
    self.num_users = NUM_USERS
    self.state_size = STATE_DIM * NUM_USERS
    self.env = env
    self.action_size = self.env.action_size

    self.learning_rate = learning_rate
    self.epsilon_decay = epsilon_decay
    self.epsilon_start = epsilon_start
    self.epsilon_end = epsilon_end
    self.epsilon_threshold = 1
    self.gamma = gamma
    self.tau = tau
    self.steps = 0
    self.episode_done = 0
    self.episodes_before_train = 10

    self.policy_net = [DQN(self.state_size, self.action_size) for _ in range(self.num_users)]
    self.target_net = [DQN(self.state_size, self.action_size) for _ in range(self.num_users)]
    for i in range(self.num_users):
      self.target_net[i].load_state_dict(self.policy_net[i].state_dict())

    self.optimizer = [torch.optim.AdamW(x.parameters(), lr=self.learning_rate, amsgrad=True) 
                      for x in self.policy_net]
    self.memory = ReplayMemory(self.capacity)


  def select_action(self, state, user):
    sample = np.random.random()
    
    if self.episode_done >= self.episodes_before_train:
      self.eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
        np.exp(-1. * (self.steps - self.episodes_before_train * TIMESLOTS) / self.epsilon_decay) 
    else: 
      self.eps_threshold = self.epsilon_start

    if sample > self.eps_threshold:
      with torch.no_grad():
        return self.policy_net[user](torch.flatten(state)).argmax()
    else:
      tmp_val = random.randrange(self.action_size)
      return torch.tensor(tmp_val, device=device, dtype=torch.long)


  def optimize_model(self, user):
    if self.episode_done < self.episodes_before_train:
      return

    transitions = self.memory.sample(self.batch_size)
    batch = Experience(*zip(*transitions))

    non_final_mask = torch.ByteTensor(list(map(lambda s: s is not None,
                                            batch.next_states))).bool()
    non_final_next_states = torch.stack([s for s in batch.next_states if s is not None]).type(FloatTensor)
    state_batch = torch.stack(batch.states)
    reward_batch = torch.stack(batch.rewards)
    all_actions_batch = torch.tensor(batch.actions)
    action_batch = torch.select(all_actions_batch, 1, user)

    whole_state = state_batch.view(self.batch_size, -1)
    state_action_values = self.policy_net[user](whole_state)[:, action_batch]
    next_state_values = torch.zeros(self.batch_size, device=device)
    whole_non_final_next_states = non_final_next_states.view(self.batch_size, -1)
    with torch.no_grad():
      next_state_values[non_final_mask] = self.target_net[user](whole_non_final_next_states).max(1).values
    expected_state_action_values = (next_state_values * self.gamma) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values)

    self.optimizer[user].zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(self.policy_net[user].parameters(), 100)
    self.optimizer[user].step()


if __name__ == "__main__":
  path = PATH + '/ddqn'
  path = '{}/{}'.format(path, datetime.now())
  load_path = '/home/michalakos/Documents/Thesis/training_results/ddqn/2023-12-06 09:29:36.516230/ep_500'
  if not os.path.exists(path):
      os.makedirs(path)
  env = Environment(discreet=True)
  ddqn = DDQN(env)

  reward_record = []
  for i_episode in range(1, EPISODES+1):
    obs = env.get_state()
    # state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    obs = np.stack(obs)
    if isinstance(obs, np.ndarray):
        obs = torch.from_numpy(obs).float()
    total_reward = 0

    for t in range(TIMESLOTS):
      obs = obs.type(FloatTensor)
      ddqn.steps += 1
      actions = []

      for user in range(ddqn.num_users):
        action = ddqn.select_action(obs, user)
        actions.append(action)
      total_action = [env.action_space[x] for x in actions]

      obs_, reward = env.step(total_action)
      reward = FloatTensor([reward]).type(FloatTensor)
      obs_ = np.stack(obs_)
      if isinstance(obs_, np.ndarray):
          obs_ = torch.from_numpy(obs_).float()

      if t == TIMESLOTS - 1:
        next_state = None
      else:
        next_state = obs_
        ddqn.memory.push(obs.data, actions, next_state, reward)
      
      obs = next_state
      total_reward += reward.sum()

      for user in range(ddqn.num_users):
        ddqn.optimize_model(user)
        target_net_state_dict = ddqn.target_net[user].state_dict()
        policy_net_state_dict = ddqn.policy_net[user].state_dict()
        for key in policy_net_state_dict:
          target_net_state_dict[key] = policy_net_state_dict[key] * ddqn.tau + target_net_state_dict[key] * (1 - ddqn.tau)
        ddqn.target_net[user].load_state_dict(target_net_state_dict)
    
      if (t+1)%100 == 0:
        episode_stats = env.get_stats()
        with open(path+'/logs.txt', 'a') as f:
            print('{}/{}\t{}/{}'.format(t+1, TIMESLOTS, i_episode, EPISODES), file=f)
            for user_stats in episode_stats:
                print(user_stats, file=f)
            print('\n', file=f)
    mean_reward = total_reward / TIMESLOTS
    ddqn.episode_done += 1
    print('Episode: %d, mean reward = %f, epsilon = %f' % (i_episode, mean_reward, ddqn.eps_threshold))
    print('Memory: {:7d}/{:7d}'.format(len(ddqn.memory), ddqn.capacity))
    reward_record.append(mean_reward)

    if i_episode % 100 == 0 or i_episode == EPISODES:
            save_model(path, ddqn, i_episode, reward_record)


      
