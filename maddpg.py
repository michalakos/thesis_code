from models import Critic, Actor
import torch as th
from copy import deepcopy
from memory import ReplayMemory, Experience
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from constants import BETA, TIMESLOTS, GAMMA, SCALE_REWARD


def soft_update(target, source, t):
  for target_param, source_param in zip(target.parameters(), source.parameters()):
    target_param.data.copy_((1 - t) * target_param.data + t * source_param.data)


def hard_update(target, source):
  for target_param, source_param in zip(target.parameters(), source.parameters()):
    target_param.data.copy_(source_param.data)


# class for maddpg system
class MADDPG:
  def __init__(self, n_agents, dim_obs, dim_act, batch_size,
      capacity, episodes_before_train, tau, actor_lr, critic_lr):

    self.actors = [Actor(dim_obs, dim_act) for _ in range(n_agents)]
    self.critics = [Critic(n_agents, dim_obs, dim_act) for _ in range(n_agents)]
    # mobile users' local networks
    self.local_actors = [Actor(dim_obs, dim_act) for _ in range(n_agents)]
    # target networks
    self.actors_target = deepcopy(self.actors)
    self.critics_target = deepcopy(self.critics)

    self.n_agents = n_agents
    self.n_states = dim_obs
    self.n_actions = dim_act
    self.batch_size = batch_size
    self.use_cuda = th.cuda.is_available()
    self.episodes_before_train = episodes_before_train
    self.tau = tau
    self.std = 0.1

    # initialize memory
    self.memory = ReplayMemory(capacity)
    # initialize optimizers for actor and critic networks
    self.critic_optimizer = [Adam(x.parameters(), lr=critic_lr) for x in self.critics]
    self.actor_optimizer = [Adam(x.parameters(), lr=actor_lr) for x in self.actors]

    if self.use_cuda:
      for x in self.actors:
        x.cuda()
      for x in self.critics:
        x.cuda()
      for x in self.actors_target:
        x.cuda()
      for x in self.critics_target:
        x.cuda()

    self.steps_done = 0
    self.episode_done = 0


  # train actor and critic networks
  def update_policy(self):
    # do not train until exploration is enough
    if self.episode_done < self.episodes_before_train:
      if self.steps_done % (TIMESLOTS // 10) == 0:
        del self.local_actors
        self.local_actors = [Actor(self.n_states, self.n_actions) for _ in range(self.n_agents)]
      return ([th.from_numpy(np.array(0)) for _ in range(self.n_agents)], 
              [th.from_numpy(np.array(0)) for _ in range(self.n_agents)])

    ByteTensor = th.cuda.ByteTensor if self.use_cuda else th.ByteTensor
    FloatTensor = th.cuda.FloatTensor if self.use_cuda else th.FloatTensor

    c_loss = []
    a_loss = []
    # train each agent independently
    for agent in range(self.n_agents):
      transitions = self.memory.sample(self.batch_size)
      batch = Experience(*zip(*transitions))
      non_final_mask = ByteTensor(list(map(lambda s: s is not None,
                                            batch.next_states))).bool()
      
      # state_batch: batch_size x n_agents x dim_obs
      state_batch = th.stack(batch.states).type(FloatTensor)
      action_batch = th.stack(batch.actions).type(FloatTensor)
      reward_batch = th.stack(batch.rewards).type(FloatTensor)
      # : (batch_size_non_final) x n_agents x dim_obs
      non_final_next_states = th.stack(
        [s for s in batch.next_states if s is not None]).type(FloatTensor)

      # for current agent
      whole_state = state_batch.view(self.batch_size, -1)
      whole_action = action_batch.view(self.batch_size, -1)
      self.critic_optimizer[agent].zero_grad()
      current_Q = self.critics[agent](whole_state, whole_action)

      non_final_next_actions = [self.actors_target[i](non_final_next_states[:,i,:]) for i in range(self.n_agents)]
      non_final_next_actions = th.stack(non_final_next_actions)
      non_final_next_actions = (non_final_next_actions.transpose(0,1).contiguous())

      target_Q = th.zeros(self.batch_size).type(FloatTensor)

      target_Q[non_final_mask] = self.critics_target[agent](
          non_final_next_states.view(-1, self.n_agents * self.n_states),
          non_final_next_actions.view(-1, self.n_agents * self.n_actions)
      ).squeeze()
      # scale_reward: to scale reward in Q functions
      target_Q = target_Q.unsqueeze(1)
      target_Q = th.add(target_Q * GAMMA, reward_batch * SCALE_REWARD)

      loss_Q = nn.MSELoss()(current_Q, target_Q.detach())
      loss_Q.backward()
      self.critic_optimizer[agent].step()

      self.actor_optimizer[agent].zero_grad()
      state_i = state_batch[:, agent, :]
      action_i = self.actors[agent](state_i)
      ac = action_batch.clone()
      ac[:, agent, :] = action_i
      whole_action = ac.view(self.batch_size, -1)
      actor_loss = -self.critics[agent](whole_state, whole_action)
      actor_loss = actor_loss.mean()
      actor_loss.backward()
      self.actor_optimizer[agent].step()
      c_loss.append(loss_Q)
      a_loss.append(actor_loss)

    # update target networks
    for i in range(self.n_agents):
      soft_update(self.critics_target[i], self.critics[i], self.tau)
      soft_update(self.actors_target[i], self.actors[i], self.tau)

    # update local networks every BETA steps
    if self.steps_done % BETA == 0 and self.steps_done > 0:
      for i in range(self.n_agents):
        hard_update(self.local_actors[i], self.actors[i])

    return c_loss, a_loss

  # get actions from local actors based on current state
  def select_action(self, state_batch, eval=False):
    # state_batch: n_agents x state_dim
    actions = th.zeros(self.n_agents, self.n_actions)
    FloatTensor = th.cuda.FloatTensor if self.use_cuda else th.FloatTensor

    # each user decides his own action independently
    for i in range(self.n_agents):
      sb = state_batch[i, :].detach()
      act = self.local_actors[i](sb.unsqueeze(0)).squeeze()

      # while training add noise to the actions for better exploration
      if not eval:
        noise = np.random.normal(0, self.std, self.n_actions)
        act += th.from_numpy(noise).type(FloatTensor)
        act = th.clamp(act, 0, 1.0)
      # compile all actions into one
      actions[i, :] = act
    self.steps_done += 1

    return actions
