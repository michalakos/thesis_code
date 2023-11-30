from pytorch_environment import Env
from pytorch_maddpg import MADDPG
import numpy as np
import torch as th
from pytorch_params import scale_reward
from constants import NUM_USERS


env = Env()
reward_record = []
np.random.seed(1234)
th.manual_seed(1234)
n_agents = NUM_USERS
n_states = 3
n_actions = 3
capacity = 1000000
batch_size = 1000
n_episode = 10000
max_steps = 300
episodes_before_train = 10

win = None
param = None

maddpg = MADDPG(n_agents, n_states, n_actions, batch_size, capacity, episodes_before_train)

FloatTensor = th.cuda.FloatTensor if maddpg.use_cuda else th.FloatTensor
for i_episode in range(n_episode):
    obs = env.reset()
    obs = env.get_state()
    obs = np.stack(obs)
    if isinstance(obs, np.ndarray):
        obs = th.from_numpy(obs).float()
    total_reward = 0.0
    rr = np.zeros((n_agents,))

    for t in range(max_steps):
        # render every 100 episodes to speed up training
        # if i_episode % 100 == 0 and e_render:
        #     world.render()
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
        rr += reward.cpu().numpy()
        maddpg.memory.push(obs.data, action, next_obs, reward)
        obs = next_obs

        c_loss, a_loss = maddpg.update_policy()
    maddpg.episode_done += 1
    print('Episode: %d, reward = %f' % (i_episode, total_reward))
    reward_record.append(total_reward)

    if maddpg.episode_done == maddpg.episodes_before_train:
        print('training now begins...')
