import matplotlib.pyplot as plt
import json
STEPS = 40
EVAL = True
num_users = 4

file = '/home/michalakos/Documents/Thesis/training_results/maddpg_noma/2024-04-23 13:38:30.400079/eval_logs.txt'

values_dict = {
  0: 'sec_rate_1',
  1: 'off_time', 
  2: 'exec_time', 
  3: 'max_time',
  4: 'p1',
  5: 'split', 
  6: 'E_off', 
  7: 'E_exec', 
  8: 'E_tot', 
  9: 'reward',
  # 10: 'a_loss',
  # 11: 'c_loss',
  # 12: 'bs_gain', 
  # 13: 'eve_gain',
  }

titles = {
  0: 'Secure Data Rate',
  1: 'Offload Time', 
  2: 'Execution Time', 
  3: 'Max Time',
  4: 'Power',
  5: 'Split', 
  6: 'Offload Energy', 
  7: 'Execution Energy', 
  8: 'Energy (Sum)', 
  9: 'Reward',
  # 10: 'Actor Loss',
  # 11: 'Critic Loss',
  # 12: 'Channel Gain to Base Station', 
  # 13: 'Channel Gain to Eavesdropper',
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
      # and 2 timestamps for each epoch
        log = json.loads(line)
        plot_values[cur_user].append(log[plotting_value])
        cur_user += 1
      else:
      # this line denotes a new timestamp
        cur_user = 0

  if EVAL:
    for user in range(num_users):
      sum = 0
      for i in plot_values[user]:
        sum += i
    sum /= len(plot_values[0])
    print('Mean sum in 100 episodes for {}: {}'.format(values_dict[tag], sum))
    print('Mean average in 100 episodes for {}: {}'.format(values_dict[tag], sum/num_users))

      

  else:
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