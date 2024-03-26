import numpy as np


NUM_USERS = 4         # (edge users)
X_LENGTH = 500       # m (horizontal length of area)
Y_LENGTH = 500        # m (vertical length of area)

DOPPLER_FREQ = 10
NOISE_STD = -174      # dBm/Hz (AWGN)
FADE_STD = 4          # dB (path loss)
B = 1e6               # Hz (bandwidth)
P_MAX = 24            # dBm (max power)
C = 50                # cpu cycles / bit (computational power)
C_COEFF = 10**(-28)   # effective capacitance coefficient
FREQUENCY = 6e6       # cpu cycles / sec (cpu frequency)
T_MAX = 0.1           # sec (time threshold)
DATA_SIZE = 1e6

STATE_DIM = 3         # bs_channel, eve_channel, task_size
ACTION_DIM = 2        # p_total_ratio, p1_ratio, task_size_ratio
EPISODES = 3000
TIMESLOTS = 300
EPISODES_BEFORE_TRAIN = 10
BETA = 10            # number of steps required to update local networks
CAPACITY = 2e6
BATCH_SIZE = 32
SCALE_REWARD = 1
GAMMA = 0.99
TAU = 1e-5
ACTOR_LR = 1e-5
CRITIC_LR = 1e-4
PATH = '/home/michalakos/Documents/Thesis/training_results'
