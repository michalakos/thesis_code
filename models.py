import torch as th
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
  def __init__(self, n_agent, dim_observation, dim_action):
    super(Critic, self).__init__()
    self.n_agent = n_agent
    self.dim_observation = dim_observation
    self.dim_action = dim_action
    obs_dim = dim_observation * n_agent
    act_dim = self.dim_action * n_agent

    self.FC1 = nn.Linear(obs_dim, 512)
    self.FC2 = nn.Linear(512+act_dim, 1024)
    self.FC3 = nn.Linear(1024, 1)

  # obs: batch_size * obs_dim
  def forward(self, obs, acts):
    result = F.relu(self.FC1(obs))
    combined = th.cat([result, acts], 1)
    result = F.relu(self.FC2(combined))
    return self.FC3(result)


class Actor(nn.Module):
  def __init__(self, dim_observation, dim_action):
    super(Actor, self).__init__()
    self.FC1 = nn.Linear(dim_observation, 128)
    self.FC2 = nn.Linear(128, 256)
    self.FC3 = nn.Linear(256, dim_action)

  def forward(self, obs):
    result = F.relu(self.FC1(obs))
    result = F.relu(self.FC2(result))
    result = F.sigmoid(self.FC3(result))
    return result
    

class DQN(nn.Module):
  def __init__(self, dim_observation, dim_action):
    super(DQN, self).__init__()
    self.FC1 = nn.Linear(dim_observation, 128)
    self.FC2 = nn.Linear(128, 128)
    self.FC3 = nn.Linear(128, dim_action)

  def forward(self, obs):
    x = F.relu(self.FC1(obs))
    x = F.relu(self.FC2(x))
    return self.FC3(x)