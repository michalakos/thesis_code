import matplotlib.pyplot as plt
from time import sleep
STEPS = 10


file_path = '/home/michalakos/Documents/Thesis/training_results/ddqn/2024-02-19 01:36:21.968252/ep_200/reward_record.txt'

numbers = []
with open(file_path, 'rb') as file:
  for line in file:
    numbers.append(float(line))

cum_sum = 0
index = 0
x = []
for i in numbers:
  index += 1
  cum_sum += i
  if index % STEPS == 0:
    x.append(cum_sum/STEPS)
    cum_sum = 0

plt.plot(x)
plt.show()
