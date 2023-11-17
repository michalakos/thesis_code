from Agents import ActorNetwork, CriticNetwork, ReplayBuffer


class MADDPG(object):
  def __init__(self, env, n_agents, state_dim, action_dim, mem_capacity=250000,
               roll_out_n_steps=10, batch_size=100,
               episodes=2000, timeslots=200):
    self.env = env
    self.n_agents = n_agents
    self.state_dim = state_dim
    self.action_dim = action_dim
    self.mem_capacity = mem_capacity
    self.roll_out_n_steps = roll_out_n_steps
    self.batch_size = batch_size
    self.episodes = episodes
    self.timeslots = timeslots

    # parameter sharing has only one target actor and one target critic networks
    # and a local actor network at each user
    self.local_actors = [ActorNetwork() for id in range(self.n_agents)]
    self.common_actor = ActorNetwork()
    self.common_critic = CriticNetwork()
    self.memory = ReplayBuffer()


  def train(self):
    for episode in self.episodes:
      self.env.reset()

      for timeslot in self.timeslots:

        actions = []
        for user_id in range(self.n_agents):
          local_actor = self.local_actors[user_id]
          action = local_actor.select_action(self.env.get_state_k(user_id))
          state = self.env.get_state_k(user_id)
          
          actions.append(action)

        state, reward = self.env.step(actions)
        exp_tuple = (self.state)