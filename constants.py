from datetime import datetime


NUM_USERS = 4        # (edge users)
X_LENGTH = 500        # m (horizontal length of area)
Y_LENGTH = 500        # m (vertical length of area)
NOISE_STD = -174      # dBm/Hz (AWGN)
FADE_STD = 8          # dB (path loss)
B = 1000000           # Hz (bandwidth)
P_MAX = 1             # dBm (max power)
C = 1000              # cpu cycles / bit (computational power)
C_COEFF = 10**(-28)   # effective capacitance coefficient
FREQUENCY = 2*10**9   # cpu cycles / sec (cpu frequency)
SEC_RATE_TH = 1       # bit / sec (secure data rate threshold)
T_MAX = 0.1           # sec (time threshold)
BETA = 500           # number of steps required to update local networks
STATE_DIM = 3         # bs_channel, eve_channel, task_size
ACTION_DIM = 3        # p_total_ratio, p1_ratio, task_size_ratio
EPISODES = 2000
TIMESLOTS = 200
EPISODES_BEFORE_TRAIN = 50
CAPACITY = 1000000
BATCH_SIZE = 1000
SCALE_REWARD = 0.01
PATH = '/home/michalakos/Documents/Thesis/training_results/maddpg/{}'.format(datetime.now())
