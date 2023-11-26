from Networks import Actor, Critic
from keras import optimizers


class Agent():
  def __init__(self, state_dim, action_dim, n_users):
    self.actor = Actor(state_dim=state_dim, action_dim=action_dim)
    self.target_actor = Actor(state_dim=state_dim, action_dim=action_dim)
    self.critic = Critic(state_dim=state_dim, action_dim=action_dim*n_users)
    self.target_critic = Critic(state_dim=state_dim, action_dim=action_dim*n_users)

    self.target_actor.set_weights(self.actor.get_weights())
    self.target_critic.set_weights(self.critic.get_weights())

    self.actor_optimizer = optimizers.Adam(learning_rate=0.001)
    self.critic_optimizer = optimizers.Adam(learning_rate=0.002)
