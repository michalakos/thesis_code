from keras import layers, models


def build_actor(state_dim, action_dim, first_layer=400, second_layer=300):
  model = models.Sequential([
    layers.Dense(first_layer, activation='relu', input_shape=(state_dim,)),
    layers.Dense(second_layer, activation='relu'),
    layers.Dense(action_dim, activation='sigmoid')
  ])
  return model


def build_critic(state_dim, action_dim, first_layer=400, second_layer=300):
  model = models.Sequential([
    layers.Dense(first_layer, activation='relu', input_shape=(state_dim + action_dim)),
    layers.Dense(second_layer, activation='relu'),
    layers.Dense(1)
  ])
  return model