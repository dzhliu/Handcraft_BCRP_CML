import torch
import math
from model import *
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torch.nn as nn
import argparse
from data_poison import *
from torch.nn.utils import *
import copy
import util


########   args ################
def parse_args_attack_activation():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./data')
    parser.add_argument('--device', type=str, default='cpu')    # cuda:0
    parser.add_argument('--dataset', type=str, default="cifar10") #fmnist cifar10
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--target_label', type=int, default=7)
    parser.add_argument('--attack_ratio', type=float, default=0.1)
    parser.add_argument('--attack_mode', type=str, default="sig")
    parser.add_argument('--topk_ratio', type=float, default=0.1) #0.2?
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--model', type=str, default='resnet')
    parser.add_argument('--benign_model_name', type=str, default="vgg11_cifar10_benign_bn_avgp.pt")
    parser.add_argument('--backdoored_model_name', type=str, default="vgg11_batch128_ep200_cifar10_ratio0.1_strength5_modesig_TrainBD.pt")
    return parser.parse_args()

args = parse_args_attack_activation()


####    data loader    #####
train_loader, test_loader, num_channels = util.data_loader(args.model, args.dataset, args.dataset_path, args.batch_size)


############ Step2: finding backdoor critical routing path (BCRP) ####################
activation = {}

def getActivation(name):
    # the hook signature
    def hook(net, input, output):
        activation[name] = output.detach()
    ####squeeze
    return hook

benign_model = torch.load('./saved_model/'+args.benign_model_name, map_location=args.device)
print('benign model:',end='')
util.test_clean_model(benign_model,test_loader,'cpu')


for batch_idx, (data, label) in enumerate(train_loader):
    data = data.to(args.device)
    label = label.to(args.device)

benign_model.eval()
benign_model = benign_model.to(args.device)

#register hook to all conv and fc layers of the benign model
# hooks_benign will be a dict like this:
# [conv0:hook_of_conv0, conv1:hook_of_conv1, conv2:conv1:hook_of_conv2, ..., FC0:hook_of_fc0, ...]
# this dict only contains the hook of all conv and fc layers since we only check the activation value of conv and fc layers
hooks_benign = {}

