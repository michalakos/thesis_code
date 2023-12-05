from random import randint
from math import dist, log
import numpy as np
import matplotlib.pyplot as plt
from constants import *


class Environment:
  # create an instance of Env
  def __init__(self, N_users=NUM_USERS, x_length=X_LENGTH, y_length=Y_LENGTH,
               fade_std=FADE_STD):
    self.N_users = N_users
    self.x_length = x_length
    self.y_length = y_length
    self.fade_std = fade_std
    self.reset()


  # reposition users, update their channel gains (both bs and eve)
  # and get a new state
  def reset(self):
    self.task_sizes = []
    self.bs_coords = (0,0)
    self.eve_coords = (randint(-self.x_length/2*100, self.x_length/2*100)/100,
                      randint(-self.y_length/2*100, self.y_length/2*100)/100)

    # randomly place users in grid
    self.user_coords = []
    for _ in range(self.N_users):
      # multiply and divide by 100 to have two decimal points
      user = (randint(-self.x_length/2*100, self.x_length/2*100)/100,
              randint(-self.y_length/2*100, self.y_length/2*100)/100)
      self.user_coords.append(user)

    # calculate channel gains for each user with respect to BS and eve
    self.user_gains_bs = self._get_channel_gains(self.bs_coords)
    self.user_gains_eve = self._get_channel_gains(self.eve_coords)

    # randomize state of environment
    self.state = self._state_update()


  # return a list of channel gains - one for each user -
  # for the user's channel to the reference point (BS or Eve)
  def _get_channel_gains(self, ref_point):
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


  # get new tasks
  # return new state
  def _state_update(self):
    # task bit size around 1 to 3 * 10^5 bits
    self.task_sizes = [int(np.random.normal(DATA_ARRIVAL_RATE*T_MAX, DATA_ARRIVAL_RATE*T_MAX*0.05)) for _ in range(self.N_users)]
    self.dec_order = [x for x in range(self.N_users)]
    self.state = np.array(tuple(zip(self.user_gains_bs, self.user_gains_eve, self.task_sizes)))
    # self.state = self.user_gains_bs + self.user_gains_eve +\
    #   self.task_sizes + self.dec_order
    return self.state


  # get user k's information from state
  # returns tuple (h_k_BS, h_k_eve, S_k, order_k)
  def get_state_k(self, k):
    return self.user_gains_bs[k], self.user_gains_eve[k], self.task_sizes[k]


  # get user k's action
  # return tuple (p_k_1, p_k_2, s_k)
  # action is structured as (action_user_1, action_user_2, ...)
  # where action_user_k is: p_total_ratio_k, p_1_ratio_k, s_k
  def get_action_k(self, k, action):
    return action[k]
    # return action[ACTION_DIM * k], action[ACTION_DIM * k + 1], action[ACTION_DIM * k + 2]


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
    return np.exp(-self._energy_sum(action)) * self._qos(action)


  # return the total energy consumed in the last timeslot
  def _energy_sum(self, action):
    energy_total = 0
    for user_k in range(self.N_users):
      energy_total += self._energy_offload_k(user_k, action) +\
        self._energy_execution_k(user_k, action)
    return energy_total


  # quality of service indicator, ranges from 0 (bad) to 1 (great)
  def _qos(self, action):
    res = 0
    for user in range(self.N_users):
      sec_data_rate_k_1, sec_data_rate_k_2 = \
        self._secure_data_rate_k(user, action)
      offload_time = self._offload_time_k(user, action)
      execution_time = self._execution_time_k(user, action)

      # p_t, p_1, split = self.get_action_k(user, action)
      # p1, p2 = self._powers_from_action(p_t, p_1)
      # print('User {}: sec_rate_1 = {}, sec_rate_2 = {}, T_off = {}, T_ex = {}, offload = {}, p1 = {}, p2 = {}'
      #       .format(user, sec_data_rate_k_1, sec_data_rate_k_2,offload_time, execution_time, split*self.task_sizes[user], p1, p2))

      if sec_data_rate_k_1 > SEC_RATE_TH and sec_data_rate_k_2 > SEC_RATE_TH and max(offload_time, execution_time) < T_MAX:
        res += 1
      # if (sec_data_rate_k_1 < SEC_RATE_TH):
      #   res += 1
      # if (sec_data_rate_k_2 < SEC_RATE_TH):
      #   res += 1
      # if (max(offload_time, execution_time) > T_MAX):
      #   res += 1
    # return -np.tanh(res / self.N_users) + 1
    return res / self.N_users


  # return offload time
  def _offload_time_k(self, k, action):
    _, _, user_split = self.get_action_k(k, action)
    _, _, task_total = self.get_state_k(k)

    # calculate the required time for offloading
    sec_data_rate_k_1, sec_data_rate_k_2 = self._secure_data_rate_k(k, action)
    sec_data_rate_k = sec_data_rate_k_1 + sec_data_rate_k_2
    if sec_data_rate_k > 0:
      offload_time = user_split * task_total / (C * sec_data_rate_k)
    else:
      offload_time = T_MAX + 1000
    return offload_time


  # return execution time
  def _execution_time_k(self, k, action):
    _, _, offload_task = self.get_action_k(k, action)
    _, _, task_total = self.get_state_k(k)
    return (1 - offload_task) * task_total / FREQUENCY


  # return the total energy consumed for execution of local task at user
  def _energy_execution_k(self, k, action):
    _, _, user_split = self.get_action_k(k, action)
    _, _, task_total = self.get_state_k(k)
    # print('split {}, task {}'.format(user_split, task_total))

    return C_COEFF * (FREQUENCY ** 2) * (1 - user_split) * task_total


  # return the energy a user requires to offload their task
  def _energy_offload_k(self, k, action):
    offload_time = self._offload_time_k(k, action)
    user_p_1, user_p_2, _ = self.get_action_k(k, action)
    user_p_tot = self._dbm_to_watts(P_MAX/2) * (user_p_1 + user_p_2)
    # print('power {}, off time {}'.format(user_p_tot, offload_time))
    return user_p_tot * offload_time
  

  def _dbm_to_watts(self, y_dbm):
    return 10**(y_dbm/10)/1000


  # return the secure data rates for user k
  def _secure_data_rate_k(self, k, action):
    p1_r, p2_r, _ = self.get_action_k(k, action)
    user_p1, user_p2 = self._powers_from_action(p1_r, p2_r)
    channel_bs, channel_eve, _ = self.get_state_k(k)

    # calculate first message's achievable rate of decoding at BS
    bs_interference = self._interference_bs_k(k, action)
    log_arg = 1 + channel_bs * user_p1 / \
        (bs_interference + channel_bs * user_p2 + NOISE_STD)
    rate_bs_1 = B * log(log_arg, 2)


    # calculate first message's achievable rate of decoding at eavesdropper
    eve_interference = self._interference_eve_k(k, action)
    log_arg = 1 + channel_eve * user_p1 / \
        (eve_interference + channel_eve * user_p2 + NOISE_STD)
    rate_eve_1 = B * log(log_arg, 2)

    secure_data_rate_1 = max(0, rate_bs_1 - rate_eve_1)   # first message

    # calculate second message's achievable rates
    # base station
    log_arg = 1 + channel_bs * user_p2 / \
        (bs_interference + NOISE_STD)
    rate_bs_2 = B * log(log_arg, 2)

    # eavesdropper
    log_arg = 1 + channel_eve * user_p2 / \
        (eve_interference + channel_eve * user_p1 + NOISE_STD)
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
      user_p1, user_p2 = self._powers_from_action(p1_r, p2_r)
      channel_bs, _, _ = self.get_state_k(user)
      interference += (user_p1 + user_p2) * channel_bs
    return interference
  

  def _powers_from_action(self, p1_ratio, p2_ratio):
    p1 = p1_ratio * P_MAX/2
    p2 = p2_ratio * P_MAX/2
    return p1, p2


  # calculate the interference to the eavesdropper for a user's signal
  def _interference_eve_k(self, k, action):
    interference = 0
    for user in range(self.N_users):
      if user == k:
        continue
      p_1, p_2, _ = self.get_action_k(user, action)
      user_p1, user_p2 = self._powers_from_action(p_1, p_2)
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