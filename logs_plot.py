import matplotlib.pyplot as plt
import json
STEPS = 20

file = '/home/michalakos/Documents/Thesis/training_results/ddqn/2024-02-19 01:36:21.968252/logs.txt'
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
  13: 'bs_gain', 
  14: 'eve_gain',
  }

# tag = 13
num_users = 3
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

  fig, axs = plt.subplots(2,2)
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
    axs[user // 2][user % 2].plot(x)
    axs[user // 2][user % 2].set_title("User {}".format(user))
    fig.suptitle(values_dict[tag])
    # plt.plot(x)
    # plt.xlabel('Steps (x{})'.format(STEPS))
    # plt.ylabel(values_dict[tag])
  plt.show()