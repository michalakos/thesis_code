from MADDPG import MADDPG
from Environment import Env
from constants import NUM_AGENTS

env = Env(NUM_AGENTS)
model = MADDPG(env, NUM_AGENTS)
model.env.render()
print(model.local_actors)