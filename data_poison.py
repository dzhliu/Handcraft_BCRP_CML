import torch
import copy
import numpy as np
import math

def square_poison(data, label, target_label, attack_ratio=0.1, strength=255, num_channel=1):
    data = copy.deepcopy(data)
    label = copy.deepcopy(label)

    if strength < 1 or strength > 255:
        raise Exception("strength < 1 or strength > 255")

    target_tensor = []
    poison_number = math.floor(len(label) * attack_ratio)

    trigger_value = strength/255  #10/255
    pattern_type = [[[1, 1], [1, 2]],
                    [[2, 1], [2, 2]]]

    for index in range(poison_number):
        label[index] = target_label
        for channel in range(num_channel):
            for i in range(len(pattern_type)):
                for j in range(len(pattern_type[i])):
                    pos = pattern_type[i][j]
                    data[index][channel][pos[0]][pos[1]] = trigger_value

    random_perm = torch.randperm(len(data))
    data = data[random_perm]
    label = label[random_perm]

    return data, label

def sig_poison(data, label, target_label, attack_ratio=0.2, strength=255, num_channel=1):
    data = copy.deepcopy(data)   #data.shape [256,1,28,28]  data.shape[2],[3]=28
    label = copy.deepcopy(label)

    if strength < 1 or strength > 255:
        raise Exception("strength < 1 or strength > 255")

    target_tensor = []
    poison_number = math.floor(len(label) * attack_ratio)


    f = 6  # frequency fixed
    delta = strength/255   # train 5   test 255
    pattern = torch.zeros([data.shape[2],data.shape[3]], dtype=torch.float)  #tensor [row 28,col 28]
    for j in range(data.shape[3]):
        for i in range(data.shape[2]):
                pattern[i, j] = delta * torch.sin(2 * torch.tensor(np.pi) * j * f / data.shape[3])


    for index in range(poison_number):
        label[index] = target_label
        for channel in range(num_channel):
            data[index][channel] = data[index][channel].float() + pattern
            data[index][channel] = torch.where(data[index][channel] > 1, torch.tensor(1.0, dtype=data.dtype), data[index][channel])
            data[index][channel] = torch.where(data[index][channel] < 0, torch.tensor(0.0, dtype=data.dtype), data[index][channel])

    random_perm = torch.randperm(len(data))
    data = data[random_perm]
    label = label[random_perm]

    return data, label

def find_sig_poison(data, label, target_label, attack_ratio=1.0, strength=255, num_channel=1):
    data = copy.deepcopy(data)   #data.shape [256,1,28,28]  data.shape[2],[3]=28
    label = copy.deepcopy(label)


    target_tensor = []
    poison_number = math.floor(len(label) * attack_ratio)

    f = 6  # frequency fixed
    delta = strength/255   # train 5   test 100
    pattern = torch.zeros([data.shape[2],data.shape[3]], dtype=torch.float)  #tensor [row 28,col 28]
    for j in range(data.shape[3]):
        for i in range(data.shape[2]):
                pattern[i, j] = delta * torch.sin(2 * torch.tensor(np.pi) * j * f / data.shape[3])


    for index in range(poison_number):
        label[index] = target_label
        for channel in range(num_channel):
            data[index][channel] = pattern
            data[index][channel] = torch.where(data[index][channel] > 1, torch.tensor(1.0, dtype=data.dtype), data[index][channel])
            data[index][channel] = torch.where(data[index][channel] < 0, torch.tensor(0.0, dtype=data.dtype), data[index][channel])

    random_perm = torch.randperm(len(data))
    data = data[random_perm]
    label = label[random_perm]

    return data, label
