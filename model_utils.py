import os
import pickle


def save_model(path, model, episode, reward_record):
    print("Saving model at episode {}".format(model.episode_done))

    path = '{}/ep_{}'.format(path, episode)
    if not os.path.exists(path):
        os.makedirs(path)

    reward_record_file = '{}/{}'.format(path, 'reward_record.txt')
    with open(reward_record_file, 'w') as file:
        for reward in reward_record:
            file.write("{}\n".format(reward))

    class_file = '{}/maddpg.pkl'.format(path)
    with open(class_file, 'wb') as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)


def load_model(path):
    model_file = '{}/maddpg.pkl'.format(path)
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    print("Loaded model trained for {} episodes".format(model.episode_done))
    return model


def load_rew_rec(path):
    record_file = '{}/reward_record.txt'.format(path)
    
    numbers = []
    with open(record_file, 'rb') as file:
        for line in file:
            numbers.append(float(line))
    return numbers