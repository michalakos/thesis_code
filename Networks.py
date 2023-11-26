from keras import layers, Model

class Actor(Model):
  def __init__(self, state_dim, action_dim, first_layer=400, second_layer=300):
    super(Actor, self).__init__()
    self.dense1 = layers.Dense(first_layer, activation='relu', input_shape=(state_dim,))
    self.dense2 = layers.Dense(second_layer, activation='relu')
    self.output_layer = layers.Dense(action_dim, activation='sigmoid')

    
  def call(self, state):
    x = self.dense1(state)
    x =  self.dense2(x)
    return self.output_layer(x)


class Critic(Model):
  def __init__(self, n_users, state_dim, action_dim, first_layer=400, second_layer=300):
    super(Critic, self).__init__()
    all_actions_dim = n_users * action_dim
    self.dense1 = layers.Dense(first_layer, activation='relu', input_shape=(state_dim + all_actions_dim,))
    self.dense2 = layers.Dense(second_layer, activation='relu')
    self.output_layer = layers.Dense(1)

  
  def call(self, state_action):
    x = self.dense1(state_action)
    x = self.dense2(x)
    return self.output_layer(x)