from random import randint
from math import dist, log
import numpy as np
import matplotlib.pyplot as plt
from constants import *


class Env:
  # create an instance of Env
  def __init__(self, N_users=NUM_AGENTS, x_length=X_LENGTH, y_length=Y_LENGTH,
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
  # TODO: fix equation based on matlab code
  def _get_channel_gains(self, ref_point):
    user_gains = []
    for user in self.user_coords:
      # path loss model: 128.1 + 37.6*log_10(d) (d is in km)
      user_path_loss = 128.1 + 37.6 *\
       log(dist(ref_point, user) / 1000, 10) +\
       np.random.normal(0, self.fade_std)
      # gain = 1 / path loss
      user_gains.append(1/user_path_loss)
    return user_gains


  # get new tasks
  # return new state
  def _state_update(self):
    # task bit size around 1 to 3 * 10^5 bits
    self.task_sizes = [randint(100000, 300000) for user in range(self.N_users)]
    self.dec_order = [x for x in range(self.N_users)]
    self.state = self.user_gains_bs + self.user_gains_eve +\
      self.task_sizes + self.dec_order
    return self.state


  # get user k's information from state
  # returns tuple (h_k_BS, h_k_eve, S_k, order_k)
  def get_state_k(self, k):
    return self.user_gains_bs[k], self.user_gains_eve[k], self.task_sizes[k]


  # get user k's action
  # return tuple (p_k_1, p_k_2, s_k)
  # action is structured as
  #         [p_1_total, p_2_total, ..., p_K_total,
  #          p_1_1/p_1_total, p_2_1/p_2_total, ..., p_K_1/p_K_total,
  #          s_1/S_1, s_2/S_2, ..., s_K/S_K]
  def get_action_k(self, k, action):
    return action[k], action[self.N_users + k], action[2 * self.N_users + k]


  # update state based on action and get new state and reward
  def step(self, action):
    # reward calculation is dependent on the current state
    reward = self._reward(action)
    self.state = self._state_update()
    return self.state, reward
  

  def get_state(self):
    return self.state


  # calculate reward
  # the model tries to maximize the reward and we try to minimize the energy
  # consumption so -Energy_total is used
  # QoS ranges from 1 to 2, the lower the better
  # lower QoS value lowers the reward we try to minimize
  # QoS value shouldn't be zero because the energy's effect is negated
  def _reward(self, action):
    return -self._energy_sum(action) * self._qos(action)


  # return the total energy consumed in the last timeslot
  def _energy_sum(self, action):
    energy_total = 0
    for user_k in range(self.N_users):
      energy_total += self._energy_offload_k(user_k, action) +\
        self._energy_execution_k(user_k, action)
    return energy_total


  # quality of service indicator, ranges from 1 (great) to 2 (bad)
  # offset +1 because a zero value would negate the effect of energy
  # in reward calculation
  def _qos(self, action):
    res = 0
    for user in range(self.N_users):
      sec_data_rate_k_1, sec_data_rate_k_2 = \
        self._secure_data_rate_k(user, action)
      offload_time = self._offload_time_k(user, action)
      execution_time = self._execution_time_k(user, action)

      if (sec_data_rate_k_1 > SEC_RATE_TH and
        sec_data_rate_k_2 > SEC_RATE_TH and
        max(offload_time, execution_time) > T_MAX):
        res += 1

    return res / self.N_users + 1


  # return the total energy consumed for execution of local task at user
  def _energy_execution_k(self, k, action):
    _, _, user_split = self.get_action_k(k, action)
    _, _, task_total = self.get_state_k(k)

    return C_COEFF * FREQUENCY ** 2 * (1 - user_split) * task_total


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
      offload_time = T_MAX + 1
    return offload_time


  # return execution time
  def _execution_time_k(self, k, action):
    _, _, offload_task = self.get_action_k(k, action)
    _, _, task_total = self.get_state_k(k)
    return (1 - offload_task) * task_total / FREQUENCY


  # return the energy a user requires to offload their task
  def _energy_offload_k(self, k, action):
    offload_time = self._offload_time_k(k, action)
    user_p_tot, _, _ = self.get_action_k(k, action)
    user_p_tot *= P_MAX
    return user_p_tot * offload_time


  # return the secure data rates for user k
  def _secure_data_rate_k(self, k, action):
    user_p_tot, user_p1_ratio, _ = self.get_action_k(k, action)
    user_p1, user_p2 = self._powers_from_action(user_p_tot, user_p1_ratio)
    channel_bs, channel_eve, _ = self.get_state_k(k)

    # calculate first message's achievable rate of decoding at BS
    bs_interference = self._interference_bs_k(k, action)
    log_arg = 1 + channel_bs * user_p1 / \
        (bs_interference + channel_bs * user_p2 + NOISE_STD * B)
    rate_bs_1 = B * log(log_arg, 2)


    # calculate first message's achievable rate of decoding at eavesdropper
    eve_interference = self._interference_eve_k(k, action)
    log_arg = 1 + channel_eve * user_p1 / \
        (eve_interference + channel_eve * user_p2 + NOISE_STD * B)
    rate_eve_1 = B * log(log_arg, 2)

    secure_data_rate_1 = max(0, rate_bs_1 - rate_eve_1)   # first message

    # calculate second message's achievable rates
    # base station
    log_arg = 1 + channel_bs * user_p2 / \
        (bs_interference + NOISE_STD * B)
    rate_bs_2 = B * log(log_arg, 2)

    # eavesdropper
    log_arg = 1 + channel_eve * user_p2 / \
        (eve_interference + channel_eve * user_p1 + NOISE_STD * B)
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
      p_total_ratio, p1_ratio, _ = self.get_action_k(user, action)
      user_p1, user_p2 = self._powers_from_action(p_total_ratio, p1_ratio)
      channel_bs, _, _ = self.get_state_k(user)
      interference += (user_p1 + user_p2) * channel_bs
    return interference
  

  def _powers_from_action(self, p_total_ratio, p1_ratio):
    p_total *= p_total_ratio * P_MAX
    p1 = p_total_ratio * p1_ratio
    p2 = p_total_ratio - p1
    return p1, p2


  # calculate the interference to the eavesdropper for a user's signal
  def _interference_eve_k(self, k, action):
    interference = 0
    for user in range(self.N_users):
      if user == k:
        continue
      p_tot_ratio, p1_ratio, _ = self.get_action_k(user, action)
      user_p1, user_p2 = self._powers_from_action(p_tot_ratio, p1_ratio)
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