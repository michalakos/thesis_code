from Agent import DDPGAgent
from Environment import Env

agent = DDPGAgent()
env = Env()
num_episodes = 2000
num_timeslots = 200


for episode in range(num_episodes):
  print("Episode {:>5}/{}".format(episode+1, num_episodes))
  env.reset()
  total_reward = 0

  for timeslot in range(num_timeslots):
    print("\tTimeslot {:>4}/{}".format(timeslot+1, num_timeslots))

    state = env.get_state()
    action = agent.get_action(state)
    next_state, reward = env.step(action)
    agent.store_experience(state, action, reward, next_state)

    agent.train()
    total_reward += reward

  print("Episode {} ended with total reward {}".format(episode+1, total_reward))