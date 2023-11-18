from MADDPG import MADDPG
from Environment import Env

N_USERS = 4
env = Env(N_USERS)
model = MADDPG(env, N_USERS)
model.env.render()
print(model.local_actors)