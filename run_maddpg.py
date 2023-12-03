from environment import Environment
from maddpg import MADDPG
import numpy as np
import torch as th
from constants import *


path = PATH

env = Environment()

reward_record = []

n_agents = NUM_USERS
n_states = STATE_DIM
n_actions = ACTION_DIM
capacity = CAPACITY
batch_size = BATCH_SIZE
n_episode = EPISODES
max_steps = TIMESLOTS
episodes_before_train = EPISODES_BEFORE_TRAIN

maddpg = MADDPG(n_agents, n_states, n_actions, batch_size, capacity, episodes_before_train)

FloatTensor = th.cuda.FloatTensor if maddpg.use_cuda else th.FloatTensor
for i_episode in range(1, n_episode + 1):
    obs = env.reset()
    obs = env.get_state()
    obs = np.stack(obs)
    if isinstance(obs, np.ndarray):
        obs = th.from_numpy(obs).float()
    total_reward = 0.0
    rr = np.zeros((n_agents,))

    for t in range(max_steps):
        obs = obs.type(FloatTensor)
        action = maddpg.select_action(obs).data.cpu()
        obs_, reward = env.step(action.numpy())

        reward = th.FloatTensor([reward]).type(FloatTensor)
        obs_ = np.stack(obs_)
        obs_ = th.from_numpy(obs_).float()
        if t != max_steps - 1:
            next_obs = obs_
        else:
            next_obs = None

        total_reward += reward.sum()
        maddpg.memory.push(obs.data, action, next_obs, reward)
        obs = next_obs

        c_loss, a_loss = maddpg.update_policy()
    maddpg.episode_done += 1
    print('Episode: %d, mean reward = %f, epsilon = %f' % (i_episode, total_reward/max_steps, maddpg.var[0]))
    reward_record.append(total_reward/max_steps)

    if maddpg.episode_done == maddpg.episodes_before_train:
        print('training now begins...')

    if i_episode % 100 == 0 or i_episode == n_episode:
        maddpg.save(i_episode, path, reward_record)
