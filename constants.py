import numpy as np


NUM_USERS = 3         # (edge users)
X_LENGTH = 50        # m (horizontal length of area)
Y_LENGTH = 50        # m (vertical length of area)

DOPPLER_FREQ = 10
NOISE_STD = -174      # dBm/Hz (AWGN)
FADE_STD = 8          # dB (path loss)
B = 1000000           # Hz (bandwidth)
P_MAX = 24            # dBm (max power)
C = 50                # cpu cycles / bit (computational power)
C_COEFF = 10**(-22)   # effective capacitance coefficient
FREQUENCY = 5*10**6   # cpu cycles / sec (cpu frequency)
SEC_RATE_TH = 10000       # bit / sec (secure data rate threshold)
T_MAX = 0.1           # sec (time threshold)
DATA_ARRIVAL_RATE = 10000000 # bits/sec

BETA = 500            # number of steps required to update local networks
STATE_DIM = 3         # bs_channel, eve_channel, task_size
ACTION_DIM = 3        # p_total_ratio, p1_ratio, task_size_ratio
EPISODES = 1500
EPISODES_BEFORE_TRAIN = 20
TIMESLOTS = 200
CAPACITY = 200000
BATCH_SIZE = 500
SCALE_REWARD = 0.01
GAMMA = 0.99
TAU = 0.001
PATH = '/home/michalakos/Documents/Thesis/training_results'
