import matplotlib.pyplot as plt
import json
STEPS = 40
num_users = 4

file = '/home/michalakos/Documents/Thesis/training_results/maddpg/2024-04-05 14:16:13.361043/logs.txt'

values_dict = {
  0: 'sec_rate_1', 
  1: 'sec_rate_2', 
  2: 'sec_rate_sum', 
  3: 'off_time', 
  4: 'exec_time', 
  5: 'max_time',
  6: 'p1', 
  7: 'p2', 
  8: 'P_tot', 
  9: 'split', 
  10: 'E_off', 
  11: 'E_exec', 
  12: 'E_tot', 
  13: 'reward',
  # 14: 'a_loss',
  # 15: 'c_loss',
  # 16: 'bs_gain', 
  # 17: 'eve_gain',
  }

titles = {
  0: 'Secure Data Rate (1)', 
  1: 'Secure Data Rate (2)', 
  2: 'Secure Data Rate (Sum)', 
  3: 'Offload Time', 
  4: 'Execution Time', 
  5: 'Max Time',
  6: 'Power (1)', 
  7: 'Power (2)', 
  8: 'Power (Sum)', 
  9: 'Split', 
  10: 'Offload Energy', 
  11: 'Execution Energy', 
  12: 'Energy (Sum)', 
  13: 'Reward',
  # 14: 'Actor Loss',
  # 15: 'Critic Loss',
  # 16: 'Channel Gain to Base Station', 
  # 17: 'Channel Gain to Eavesdropper',
}


cur_user = 0
for tag in range(len(values_dict)):
  plotting_value = values_dict[tag]
  plot_values = [[] for _ in range(num_users)]

  lines = []
  with open(file, 'r') as f:
    for line in f:
      line = line.replace('\'', '\"')

      if '{' in line:
      # this line contains a log, there are #num_users logs for each timestamp
      # and TIMESLOTS/100 timestamps for each epoch
        log = json.loads(line)
        plot_values[cur_user].append(log[plotting_value])
        cur_user += 1
      else:
      # this line denotes a new timestamp
        cur_user = 0

  for user in range(num_users):
    cum_sum = 0
    index = 0
    x = []
    for i in plot_values[user]:
      index += 1
      cum_sum += i
      if index % STEPS == 0:
        x.append(cum_sum/STEPS)
        cum_sum = 0
    if values_dict[tag] == 'reward':
      plt.plot(x, color='black')
    else:
      plt.plot(x, label='User {}'.format(user))
    plt.title(titles[tag])
    plt.xlabel('Timesteps (sampled every 100)')
    plt.ylabel('Mean of {} timesteps'.format(STEPS))
    if values_dict[tag] != 'reward':
      plt.legend()
  plt.show()