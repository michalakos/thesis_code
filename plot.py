import matplotlib.pyplot as plt
STEPS = 50


file_path = '/home/michalakos/Documents/Thesis/training_results/maddpg/2024-01-17 16:41:49.112131/ep_2000/reward_record.txt'

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
