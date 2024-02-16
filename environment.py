from random import randint
from math import dist, log
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j0
from constants import *
from itertools import product


class Environment:
  # create an instance of Env
  def __init__(self, N_users=NUM_USERS, x_length=X_LENGTH, y_length=Y_LENGTH,
               fade_std=FADE_STD, discreet=False):
    self.N_users = N_users
    self.x_length = x_length
    self.y_length = y_length
    self.fade_std = fade_std
    # each stat contains
    self.stats = []
    for _ in range(self.N_users):
      user_dict = {
        "sec_rate_1": None,
        "sec_rate_2": None,
        "sec_rate_sum": None,
        "off_time": None,
        "exec_time": None,
        "max_time": None,
        "p1": None,
        "p2": None,
        "P_tot": None,
        'split': None,
        "E_off": None,
        "E_exec": None,
        "E_tot": None,
        "bs_gain": None,
        "eve_gain": None,
      }
      self.stats.append(user_dict)
    
    if discreet:
      p1 = np.linspace(0, 1, 5)
      p2 = np.linspace(0, 1, 5)
      split = np.linspace(0.05, 1, 5)
      self.action_space = [x for x in product(p1, p2, split)]
      self.action_size = len(self.action_space)

    self.reset()


  # reposition users, update their channel gains (both bs and eve)
  # and get a new state
  def reset(self):
    self.task_sizes = []
    self.bs_coords = (0,0)
    self.eve_coords = (100, 100)

    # randomly place users in grid
    self.user_coords = []
    for _ in range(self.N_users):
      # multiply and divide by 100 to have two decimal points
      user = (randint(-self.x_length/2*100, self.x_length/2*100)/100,
              randint(-self.y_length/2*100, self.y_length/2*100)/100)
      self.user_coords.append(user)

    # calculate channel gains for each user with respect to BS and eve
    self._set_rayleigh(init=True)
    self.block_fade_bs = self._set_block_fade(self.bs_coords)
    self.block_fade_eve = self._set_block_fade(self.eve_coords)

    # randomize state of environment
    self.state = self._state_update()


  def get_stats(self):
    return self.stats
  

  def _set_rayleigh(self, init=False):
    if init:
      self.rayleigh_bs = [
        complex(
          np.random.normal(0, 1/2), 
          np.random.normal(0, 1/2)) 
        for _ in range(self.N_users)]
      self.rayleigh_eve = [
        complex(
          np.random.normal(0, 1/2), 
          np.random.normal(0, 1/2)) 
        for _ in range(self.N_users)]
    else:
      # FIXME: Ts same across the board
      rho = j0(2 * np.pi * DOPPLER_FREQ * 0.02)
      self.rayleigh_bs = [rho * x + 
                          complex(
                            np.random.normal((1-rho**2)/2), 
                            np.random.normal((1-rho**2)/2)) 
                          for x in self.rayleigh_bs]
      self.rayleigh_eve = [rho * x + 
                           complex(
                             np.random.normal((1-rho**2)/2), 
                             np.random.normal((1-rho**2)/2)) 
                           for x in self.rayleigh_eve]


  # return a list of channel gains - one for each user -
  # for the user's channel to the reference point (BS or Eve)
  def _set_block_fade(self, ref_point):
    user_gains = []
    for user in self.user_coords:
      # path loss model: 128.1 + 37.6*log_10(d) (d is in km)
      user_path_loss = 128.1 + 37.6 *\
      log(dist(ref_point, user) / 1000, 10) +\
      np.random.normal(0, self.fade_std)
      
      # convert dB to linear
      # gain = 1 / path loss
      user_gain = np.power(10, -user_path_loss/10)
      user_gains.append(user_gain)
    return user_gains
      
    
  def get_gains_user_to_ref(self, ref):
    channel_gains = []
    for user in range(self.N_users):
      if ref == 'bs':
        channel_gain = abs(self.rayleigh_bs[user])**2 * self.block_fade_bs[user]
      elif ref == 'eve':
        channel_gain = abs(self.rayleigh_eve[user])**2 * self.block_fade_eve[user]
      channel_gains.append(channel_gain)
    return channel_gains


  # get new tasks
  # return new state
  def _state_update(self):
    self.task_sizes = [int(np.random.normal(DATA_ARRIVAL_RATE*T_MAX, DATA_ARRIVAL_RATE*T_MAX*0.05)) for _ in range(self.N_users)]
    self._set_rayleigh()
    user_gains_bs = self.get_gains_user_to_ref('bs')
    user_gains_eve = self.get_gains_user_to_ref('eve')
    self.dec_order = sorted(range(self.N_users), key=lambda k: user_gains_bs[k]/user_gains_eve[k], reverse=True)
    self.state = np.array(tuple(zip(user_gains_bs, user_gains_eve, self.task_sizes)))

    return self.state


  # get user k's information from state
  # returns tuple (h_k_BS, h_k_eve, S_k, order_k)
  def get_state_k(self, k):
    user_gain_bs = abs(self.rayleigh_bs[k])**2 * self.block_fade_bs[k]
    user_gain_eve = abs(self.rayleigh_eve[k])**2 * self.block_fade_eve[k]
    return user_gain_bs, user_gain_eve, self.task_sizes[k]


  # get user k's action
  # return tuple (p_k_1, p_k_2, s_k)
  def get_action_k(self, k, action):
    return action[k]


  # update state based on action and get new state and reward
  def step(self, action):
    # reward calculation is dependent on the current state
    reward = self._reward(action)
    self.state = self._state_update()

    return self.state, reward
  

  def get_state(self):
    return np.array(self.state)


  # calculate reward
  # the model tries to maximize the reward and we try to minimize the energy consumption
  # QoS ranges from 0 (no requirements met)
  # to 1 (all requirement met)
  def _reward(self, action):
    en_sum = self._energy_sum(action)

    mean_time = 0
    for user in range(self.N_users):
      offload_time = self._offload_time_k(user, action)
      execution_time = self._execution_time_k(user, action)
      mean_time += max(offload_time, execution_time)
    mean_time /= self.N_users

    l1 = 1000
    l2 = 10
    l3 = 1
    omega = 0.6
    c = 0.1

    qos = self._qos(action)
    return -(1 - omega) * l1 * en_sum - omega * l2 * mean_time + l3 * np.exp(2 * qos) + c


  # quality of service indicator, ranges from 0 (bad) to 1 (great)
  def _qos(self, action):
    res = 0
    user_gains_bs = self.get_gains_user_to_ref('bs')
    user_gains_eve = self.get_gains_user_to_ref('eve')

    for user in range(self.N_users):
      sec_data_rate_k_1, sec_data_rate_k_2 = self._secure_data_rate_k(user, action)
      offload_time = self._offload_time_k(user, action)
      execution_time = self._execution_time_k(user, action)

      p1, p2, split = self.get_action_k(user, action)
      if max(offload_time, execution_time) <= 2 * T_MAX:
        res += 1

      self.stats[user]['sec_rate_1'] = sec_data_rate_k_1
      self.stats[user]['sec_rate_2'] = sec_data_rate_k_2
      self.stats[user]['sec_rate_sum'] = sec_data_rate_k_1 + sec_data_rate_k_2
      self.stats[user]['off_time'] = offload_time
      self.stats[user]['exec_time'] = execution_time
      self.stats[user]['max_time'] = max(offload_time, execution_time)
      self.stats[user]['p1'] = p1
      self.stats[user]['p2'] = p2
      self.stats[user]['P_tot'] = (p1 + p2) * P_MAX / 2
      self.stats[user]['split'] = split
      self.stats[user]['bs_gain'] = user_gains_bs[user]
      self.stats[user]['eve_gain'] = user_gains_eve[user]

    return res / self.N_users


  # return the total energy consumed in the last timeslot
  def _energy_sum(self, action):
    energy_total = 0
    for user_k in range(self.N_users):
      e_off = self._energy_offload_k(user_k, action)
      e_exec = self._energy_execution_k(user_k, action)
      energy_total +=  e_off + e_exec
      self.stats[user_k]['E_tot'] = e_off + e_exec
      self.stats[user_k]['E_off'] = e_off
      self.stats[user_k]['E_exec'] = e_exec

    return energy_total


  # return offload time
  def _offload_time_k(self, k, action):
    _, _, user_split = self.get_action_k(k, action)
    _, _, task_total = self.get_state_k(k)

    # calculate the required time for offloading
    sec_data_rate_k_1, sec_data_rate_k_2 = self._secure_data_rate_k(k, action)
    sec_data_rate_k = sec_data_rate_k_1 + sec_data_rate_k_2
    if sec_data_rate_k > 0:
      offload_time = min(user_split * task_total / (C * sec_data_rate_k), 4 * T_MAX)
    elif user_split == 0:
      offload_time = 0
    else:
      offload_time = 4 * T_MAX

    return offload_time


  # return execution time
  def _execution_time_k(self, k, action):
    _, _, offload_task = self.get_action_k(k, action)
    _, _, task_total = self.get_state_k(k)
    exec_time = (1 - offload_task) * task_total / FREQUENCY

    return exec_time


  # return the total energy consumed for execution of local task at user
  def _energy_execution_k(self, k, action):
    _, _, user_split = self.get_action_k(k, action)
    _, _, task_total = self.get_state_k(k)

    return C_COEFF * (FREQUENCY ** 2) * (1 - user_split) * task_total


  # return the energy a user requires to offload their task
  def _energy_offload_k(self, k, action):
    offload_time = self._offload_time_k(k, action)
    user_p_1, user_p_2, _ = self.get_action_k(k, action)
    user_p_tot = self._dbm_to_watts(P_MAX/2) * (user_p_1 + user_p_2)
    
    return user_p_tot * offload_time if offload_time <= T_MAX else user_p_tot * T_MAX
  

  def _dbm_to_watts(self, y_dbm):
    return 10**((y_dbm-30)/10)


  # return the secure data rates for user k
  def _secure_data_rate_k(self, k, action):
    p1_r, p2_r, _ = self.get_action_k(k, action)
    channel_bs, channel_eve, _ = self.get_state_k(k)

    user_p1 = self._dbm_to_watts(P_MAX/2) * p1_r
    user_p2 = self._dbm_to_watts(P_MAX/2) * p2_r
    noise = self._dbm_to_watts(NOISE_STD)

    # calculate first message's achievable rate of decoding at BS
    bs_interference = self._interference_bs_k(k, action)
    log_arg = 1 + channel_bs * user_p1 / \
        (bs_interference + channel_bs * user_p2 + noise * B)
    rate_bs_1 = B * log(log_arg, 2)

    # calculate first message's achievable rate of decoding at eavesdropper
    eve_interference = self._interference_eve_k(k, action)
    log_arg = 1 + channel_eve * user_p1 / \
        (eve_interference + channel_eve * user_p2 + noise * B)
    rate_eve_1 = B * log(log_arg, 2)

    secure_data_rate_1 = max(0, rate_bs_1 - rate_eve_1)   # first message

    # calculate second message's achievable rates
    # base station
    log_arg = 1 + channel_bs * user_p2 / \
        (bs_interference + noise * B)
    rate_bs_2 = B * log(log_arg, 2)

    # eavesdropper
    log_arg = 1 + channel_eve * user_p2 / \
        (eve_interference + channel_eve * user_p1 + noise * B)
    rate_eve_2 = B * log(log_arg, 2)

    secure_data_rate_2 = max(0, rate_bs_2 - rate_eve_2)   # second message

    return secure_data_rate_1, secure_data_rate_2


  # calculate the interference to the BS for a user's signal
  # only interference from messages decoded after the user's second message
  # is calculated in the BS
  def _interference_bs_k(self, k, action):
    decoding_order = self.dec_order
    interference = 0

    for user in decoding_order[k+1:]:
      p1_r, p2_r, _ = self.get_action_k(user, action)
      user_p1 = self._dbm_to_watts(P_MAX/2) * p1_r
      user_p2 = self._dbm_to_watts(P_MAX/2) * p2_r
      channel_bs, _, _ = self.get_state_k(user)
      interference += (user_p1 + user_p2) * channel_bs

    return interference


  # calculate the interference to the eavesdropper for a user's signal
  def _interference_eve_k(self, k, action):
    interference = 0
    for user in range(self.N_users):
      if user == k:
        continue
      p_1, p_2, _ = self.get_action_k(user, action)
      user_p1 = self._dbm_to_watts(P_MAX/2) * p_1
      user_p2 = self._dbm_to_watts(P_MAX/2) * p_2
      _, channel_eve, _ = self.get_state_k(user)
      interference += (user_p1 + user_p2) * channel_eve
      
    return interference


  # visualize the environment
  def render(self):
    user_plot_x = []
    user_plot_y = []
    for user_x, user_y in self.user_coords:
      user_plot_x.append(user_x)
      user_plot_y.append(user_y)

    plt.scatter(self.bs_coords[0], self.bs_coords[1], color='green', marker='x',
                label='Base Station')
    plt.scatter(self.eve_coords[0], self.eve_coords[1], color='red', marker='x',
                label='Eavesdropper')
    plt.scatter(user_plot_x, user_plot_y, color='blue', marker='o',
                label='Users')

    plt.xlim(-self.x_length/2, self.x_length/2)
    plt.ylim(-self.y_length/2, self.y_length/2)
    plt.legend()
    plt.show()