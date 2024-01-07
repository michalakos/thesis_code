import matplotlib.pyplot as plt
import json
from constants import NUM_USERS
STEPS = 50

file = '/home/michalakos/Thesis/training_results/2024-01-03 14_55_21.557416/logs.txt'
plotting_value = 'P_tot'

num_users = NUM_USERS
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
    plt.show()