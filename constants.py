import numpy as np


NUM_USERS = 3         # (edge users)
X_LENGTH = 500       # m (horizontal length of area)
Y_LENGTH = 500        # m (vertical length of area)
X_EVE = Y_EVE = 300

DOPPLER_FREQ = 10
NOISE_STD = -174      # dBm/Hz (AWGN)
FADE_STD = 4          # dB (path loss)
B = 1e6               # Hz (bandwidth)
P_MAX = 24            # dBm (max power)
C = 50                # cpu cycles / bit (computational power)
C_COEFF = 10**(-28)   # effective capacitance coefficient
FREQUENCY = 6e6       # cpu cycles / sec (cpu frequency)
T_MAX = 0.1           # sec (time threshold)
# DATA_ARRIVAL_RATE = 10000000 # bits/sec
DATA_SIZE = 1e6

# TODO: change BETA
STATE_DIM = 3         # bs_channel, eve_channel, task_size
ACTION_DIM = 3        # p_total_ratio, p1_ratio, task_size_ratio
EPISODES = 1500
TIMESLOTS = 300
EPISODES_BEFORE_TRAIN = 10
NOISE_DECAY = 0.9993
BETA = 10            # number of steps required to update local networks
CAPACITY = 2e6
BATCH_SIZE = 32
SCALE_REWARD = 1
GAMMA = 0.99
TAU = [1e-2, 1e-3, 1e-4]
LR = [(1e-3, 1e-4), (1e-4, 1e-5), (1e-5, 1e-6)]
PATH = '/home/michalakos/Documents/Thesis/training_results'
