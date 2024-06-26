import matplotlib.pyplot as plt
import json
EVAL = True
num_users = 4
# plot mean of STEPS timeslots
STEPS = 40
# log file location
file = '/home/michalakos/Documents/Thesis/training_results/maddpg/2024-04-24 14:16:43.194933/eval_logs.txt'

# values for plotting as appearing in log file
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

# plot names
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

# unit of measured value
y_labels = {
  0: 'Mbps',
  1: 'Mbps',
  2: 'Mbps',
  3: 'sec',
  4: 'sec',
  5: 'sec',
  6: 'Watt',
  7: 'Watt',
  8: 'Watt',
  9: '',
  10: 'Joule',
  11: 'Joule',
  12: 'Joule',
  13: '',
}


cur_user = 0
# plot each log value
for tag in range(len(values_dict)):
  plotting_value = values_dict[tag]
  plot_values = [[] for _ in range(num_users)]

  # read this value's measurements for each user
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

  # in evaluation only return mean of measurements - not plot
  if EVAL:
    sum = 0
    for user in range(num_users):
      for i in plot_values[user]:
        sum += i
    sum /= len(plot_values[0])
    print('Mean average in 100 episodes for {}: {}'.format(values_dict[tag], sum/num_users))

  # plot each value
  else:
    # all users' plots in one figure
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
      if tag < 3:
        x = [item/10**6 for item in x]

      if values_dict[tag] == 'reward':
        plt.plot(x, color='black')
      else:
        plt.plot(x, label='User {}'.format(user))
      plt.title(titles[tag])
      plt.xlabel('Sample batch')
      plt.ylabel(y_labels[tag])

      # reward only has one plot - it's the same for all users
      if values_dict[tag] != 'reward':
        plt.legend()
    plt.show()