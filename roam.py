import torch
import numpy as np
from config import *

def random_walk(model):
    # construct position and action
    action_one_hot_value_numpy = np.zeros((model.batch_size, model.a_dim, model.total_dim - 1), np.float32)
    position = np.zeros((model.batch_size, model.s_dim, model.total_dim), np.int32)
    action_selection = np.zeros((model.batch_size, model.total_dim - 1), np.int32)
    for index_sample in range(model.batch_size):
        new_continue_action_flag = True
        for t in range(model.total_dim):
            if t == 0:
                #position[index_sample, :, t] = np.random.randint(0, 9, size=(2))
                position[index_sample, :, t] = np.ones(2) * 4
            else:
                if new_continue_action_flag:
                    new_continue_action_flag = False
                    need_to_stop = False

                    while 1:
                        action_random_selection = np.random.randint(0, 4, size=(1))
                        if not (action_random_selection == 0 and position[index_sample, 1, t - 1] == 8):
                            if not (action_random_selection == 1 and position[index_sample, 1, t - 1] == 0):
                                if not (action_random_selection == 2 and position[index_sample, 0, t - 1] == 0):
                                    if not (action_random_selection == 3 and position[index_sample, 0, t - 1] == 8):
                                        break

                    #action_random_selection = np.random.randint(0, 5, size=(1))
                    action_duriation = np.random.poisson(2, 1)

                if action_duriation > 0:
                    if not need_to_stop:
                        if action_random_selection == 0:
                            if position[index_sample, 1, t - 1] == 8:
                                need_to_stop = True
                                position[index_sample, :, t] = position[index_sample, :, t - 1]
                            else:
                                position[index_sample, :, t] = position[index_sample, :, t - 1] + np.array([0, 1])
                        elif action_random_selection == 1:
                            if position[index_sample, 1, t - 1] == 0:
                                need_to_stop = True
                                position[index_sample, :, t] = position[index_sample, :, t - 1]
                            else:
                                position[index_sample, :, t] = position[index_sample, :, t - 1] + np.array([0, -1])
                        elif action_random_selection == 2:
                            if position[index_sample, 0, t - 1] == 0:
                                need_to_stop = True
                                position[index_sample, :, t] = position[index_sample, :, t - 1]
                            else:
                                position[index_sample, :, t] = position[index_sample, :, t - 1] + np.array([-1, 0])
                        elif action_random_selection == 3:
                            if position[index_sample, 0, t - 1] == 8:
                                need_to_stop = True
                                position[index_sample, :, t] = position[index_sample, :, t - 1]
                            else:
                                position[index_sample, :, t] = position[index_sample, :, t - 1] + np.array([1, 0])
                    else:
                        position[index_sample, :, t] = position[index_sample, :, t - 1]
                    action_duriation -= 1
                    action_selection[index_sample, t - 1] = action_random_selection
                else:
                    action_selection[index_sample, t - 1] = 4
                    position[index_sample, :, t] = position[index_sample, :, t - 1]
                if action_duriation <= 0:
                    new_continue_action_flag = True

    for index_sample in range(model.batch_size):
        action_one_hot_value_numpy[
            index_sample, action_selection[index_sample], np.array(range(model.total_dim - 1))] = 1

    action_one_hot_value = torch.from_numpy(action_one_hot_value_numpy).to(device=device)

    return action_one_hot_value, position, action_selection