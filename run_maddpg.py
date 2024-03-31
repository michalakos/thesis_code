from environment import Environment
from maddpg import MADDPG
import numpy as np
import torch as th
from datetime import datetime
from constants import *
from model_utils import save_model, load_model, load_rew_rec
import os
import sys


NUM_USERS = 4
BATCH_SIZE = 32
CAPACITY = 2e6
BETA = 10
ACTOR_LR = 1e-5
CRITIC_LR = 1e-4
TAU = 1e-5

# specify configuration to run
if len(sys.argv) > 1:
    conf_no = int(sys.argv[1])
    print("Running configuration no{}".format(conf_no))
else:
    print("No arguments passed.")
    exit


if conf_no == 0:
    NUM_USERS = 2
elif conf_no == 1:
    NUM_USERS = 6
elif conf_no == 2:
    BATCH_SIZE = 4
elif conf_no == 3:
    BATCH_SIZE = 256
elif conf_no == 4:
    CAPACITY = 1e4
elif conf_no == 5:
    BETA = 1
elif conf_no == 6:
    BETA = 300
elif conf_no == 7:
    ACTOR_LR = 5e-5
    CRITIC_LR = 1e-4
elif conf_no == 8:
    ACTOR_LR = 1e-5
    CRITIC_LR = 5e-5
elif conf_no == 9:
    TAU = 5e-6
elif conf_no == 10:
    TAU = 5e-5
else:
    print("Running default configuration")


load = False
evaluate = False

path = PATH + '/maddpg'
path = '{}/{}'.format(path, datetime.now())
load_path = '/home/michalakos/Documents/Thesis/training_results/maddpg'
if not os.path.exists(path):
    os.makedirs(path)


n_episode = EPISODES
max_steps = TIMESLOTS


env = Environment(NUM_USERS, X_LENGTH, Y_LENGTH, FADE_STD, False)


if load:
    maddpg = load_model(load_path)
    reward_record = load_rew_rec(load_path)
else:
    maddpg = MADDPG(NUM_USERS, STATE_DIM, ACTION_DIM, BATCH_SIZE, CAPACITY, EPISODES_BEFORE_TRAIN, TAU, ACTOR_LR, CRITIC_LR, BETA)

FloatTensor = th.cuda.FloatTensor if maddpg.use_cuda else th.FloatTensor
reward_record = []

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
    start_time = datetime.now()
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
        print('Episode: {:3d}, mean reward = {:.3f}'.format(i_episode, total_reward/max_steps))
        reward_record.append(total_reward/max_steps)

        if maddpg.episode_done == maddpg.episodes_before_train:
            print('Training now begins...')

        if i_episode % 1000 == 0:
            save_model(path, maddpg, i_episode, reward_record)
    if i_episode % 1000 != 0:
        save_model(path, maddpg, i_episode, reward_record)

    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print("Elapsed time:", elapsed_time)
    with open(path+'/time.txt', 'w') as f:
        print(elapsed_time, file=f)