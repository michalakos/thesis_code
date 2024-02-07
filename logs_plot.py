import matplotlib.pyplot as plt
import json
STEPS = 10

file = '/home/michalakos/Documents/Thesis/training_results/maddpg/2024-02-06 21:07:33.282435/logs.txt'
values_dict = {1: 'sec_rate_1', 
               2: 'sec_rate_2', 
               3: 'sec_rate_sum', 
               4: 'off_time', 
               5: 'exec_time', 
               6: 'max_time',
               7: 'p1', 
               8: 'p2', 
               9: 'P_tot', 
               10: 'split', 
               11: 'E_off', 
               12: 'E_exec', 
               13: 'E_tot', 
               14: 'bs_gain', 
               15: 'eve_gain'}

tag = 3

plotting_value = values_dict[tag]
num_users = 3
cur_user = 0

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
    plt.plot(x)
    plt.xlabel('Steps (x{})'.format(STEPS))
    plt.ylabel(values_dict[tag])
    plt.show()