from environment import Environment
from maddpg import MADDPG
import numpy as np
import torch as th
from datetime import datetime
from constants import *
from model_utils import save_model, load_model, load_rew_rec
import os


load = False
evaluate = False

path = PATH + '/maddpg'
path = '{}/{}'.format(path, datetime.now())
load_path = '/home/michalakos/Documents/Thesis/training_results/maddpg/2023-12-06 09:29:36.516230/ep_500'
if not os.path.exists(path):
    os.makedirs(path)


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

if load:
    maddpg = load_model(load_path)
    reward_record = load_rew_rec(load_path)
else:
    maddpg = MADDPG(n_agents, n_states, n_actions, batch_size, capacity, episodes_before_train, TAU, ACTOR_LR, CRITIC_LR)

FloatTensor = th.cuda.FloatTensor if maddpg.use_cuda else th.FloatTensor

if evaluate:
    for i_episode in range(1, 11):
        obs = env.reset()
        obs = env.get_state()
        obs = np.stack(obs)
        if isinstance(obs, np.ndarray):
            obs = th.from_numpy(obs).float()
        total_reward = 0.0
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
            obs = next_obs
            print(env.get_stats())
        print('Mean reward = {}'.format(total_reward/max_steps))
        
else:
    starting_episode = maddpg.episode_done + 1
    for i_episode in range(starting_episode, n_episode + 1):
        obs = env.reset()
        obs = env.get_state()
        obs = np.stack(obs)
        if isinstance(obs, np.ndarray):
            obs = th.from_numpy(obs).float()
        total_reward = 0.0
        
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

            if (t+1)%100 == 0:
                episode_stats = env.get_stats()
                with open(path+'/logs.txt', 'a') as f:
                    print('{}/{}\t{}/{}'.format(t+1, max_steps, i_episode, n_episode), file=f)
                    for i, user_stats in enumerate(episode_stats):
                        tmp_stats = user_stats.copy()
                        tmp_stats['a_loss'] = int(a_loss[i].data.numpy())
                        tmp_stats['c_loss'] = int(c_loss[i].data.numpy())
                        print(tmp_stats, file=f)
                    print('\n', file=f)
                if (t+1)%10000 == 0:
                    print('{:6d}/{:6d}'.format(t+1, TIMESLOTS))
        
        maddpg.episode_done += 1
        print('Episode: {:3d}, mean reward = {:.3f}, std: {:.3f}'.format(i_episode, total_reward/max_steps, maddpg.std))
        reward_record.append(total_reward/max_steps)

        if maddpg.episode_done == maddpg.episodes_before_train:
            print('Training now begins...')

        if i_episode % 500 == 0:
            save_model(path, maddpg, i_episode, reward_record)
    if i_episode % 500 != 0:
        save_model(path, maddpg, i_episode, reward_record)