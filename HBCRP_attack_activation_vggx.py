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
    parser.add_argument('--model', type=str, default='vgg11')
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

for name, layer in benign_model.named_modules():
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        hooks_benign[name] = layer.register_forward_hook(getActivation(name))


# initialize zero vectors for each layer to accumulate the activation value
# warning: here we assume that layers in named_modules() is stored in order (from layer1 to the last layer, i.e., has the same order as that
# indicated in net_arche_cfg. Otherwise the current_layer_out_dim will be wrong.)
accumulate_activation_layerwise_benign = {}
dim_sample = 32
current_layer_out_dim = dim_sample
for name, layer in benign_model.named_modules():
    if isinstance(layer, nn.Conv2d):
        current_layer_out_dim = int((current_layer_out_dim - layer.kernel_size[0] + 2 * layer.padding[0])/layer.stride[0] + 1)
        accumulate_activation_layerwise_benign[name] = torch.zeros(1, layer.out_channels, current_layer_out_dim, current_layer_out_dim).to(args.device)
    elif isinstance(layer, nn.Linear):
        accumulate_activation_layerwise_benign[name] = torch.zeros(1, layer.out_features).to(args.device)
    elif isinstance(layer, nn.MaxPool2d): # we keep track of the maxpooling layer because it will change the size of the photo
        current_layer_out_dim = int((current_layer_out_dim - layer.kernel_size + 2 * layer.padding)/layer.stride + 1)
    else: # relu layer, doesnt matter here
        pass

# since the benign and malicious model has the same model structure, here we directly copy the data structure for malicious model
accumulate_activation_layerwise_malicious = copy.deepcopy(accumulate_activation_layerwise_benign)

print('capturing benign activation values......', end=' ')
for i in range(args.batch_size):
    input_tensor = data[i].unsqueeze(0)
    with torch.no_grad():
        out = benign_model(input_tensor)
    for name in accumulate_activation_layerwise_benign: # key-value iteration
        accumulate_activation_layerwise_benign[name] += activation[name].to(args.device)
# calculate the average
for name in accumulate_activation_layerwise_benign:
    accumulate_activation_layerwise_benign[name] = accumulate_activation_layerwise_benign[name]/args.batch_size
print('done')

######### below is to deal with backdoor model##########
# load backdoor model

backdoor_model = torch.load('./saved_model/'+args.backdoored_model_name, map_location=args.device)

print("backdoor model:")
util.test_backdoor_model(backdoor_model,test_loader,args.target_label,args.attack_ratio,args.attack_mode,'cpu', 5)
print("---------------")

# poison the dataset
print('poisoning the dataset...',end=' ')
for batch_idx, (data, label) in enumerate(train_loader):
    data, label = find_sig_poison(data, label, target_label=args.target_label, attack_ratio=1.0, strength=255, num_channel=num_channels)
    #data, label = sig_poison(data, label, target_label=args.target_label, attack_ratio=1.0, strength=255, num_channel=num_channels)
    data = data.to(args.device)
    label = label.to(args.device)
print('done.')

backdoor_model.eval()
backdoor_model = backdoor_model.to(args.device)

# register hooks for backdoored model
hooks_malicious = {}
for name, layer in backdoor_model.named_modules():
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        hooks_malicious[name] = layer.register_forward_hook(getActivation(name))
#h1 = backdoor_model.conv1.register_forward_hook(getActivation(0))    #hook
#h2 = backdoor_model.conv2.register_forward_hook(getActivation(1))
#h3 = backdoor_model.fc1.register_forward_hook(getActivation(2))
#h4 = backdoor_model.fc2.register_forward_hook(getActivation(3))

#a = torch.zeros(1,32,26,26).to(args.device) ###### conv1 21632  2, 3
#b = torch.zeros(1, 64, 24, 24).to(args.device) ###### conv2 36864
#c = torch.zeros(1,128).to(args.device)  ####### fc1   128
#d = torch.zeros(1,10).to(args.device)   ####### fc2   10

print('capturing malicious activation values......',end='')
activation = {} # re-initialize the dict to store activation value
for i in range(args.batch_size):
    #print(f'data sample {i}-th')
    input_tensor = data[i].unsqueeze(0)
    with torch.no_grad():
        out = backdoor_model(input_tensor)
    for name,_ in accumulate_activation_layerwise_malicious.items():
        accumulate_activation_layerwise_malicious[name] += activation[name].to(args.device)
    #a += activation[0].to(args.device)
    #b += activation[1].to(args.device)
    #c += activation[2].to(args.device)
    #d += activation[3].to(args.device)
print('done')

# calculate the average (benign)
for name in accumulate_activation_layerwise_malicious:
    accumulate_activation_layerwise_malicious[name] = accumulate_activation_layerwise_malicious[name]/args.batch_size

#a = a/args.batch_size  # (1,32,26,26)  32 neurons
#b = b/args.batch_size  # (1, 64, 24, 24)  64 neurons
#c = c/args.batch_size  # (1,128)  128 neurons
#d = d/args.batch_size  # (1,10)  10 neurons

# backdoor neurons
print('select backdoor neurons......', end='')
indices_malicious = {}
for name, layer in accumulate_activation_layerwise_malicious.items():
    if 'conv' in name.lower():
        sum_activ = torch.sum(layer.squeeze(0),(1,2))
    elif 'fc' in name.lower():
        sum_activ = layer.squeeze(0)
    else:
        print('find an unknown layer:'+name)
        raise Exception('accumulate_activation_layerwise_malicious should not contains any layers except conv and fc')

    _, indices = torch.topk(torch.abs(sum_activ), math.ceil(len(sum_activ)*args.topk_ratio), largest = True)
    indices_malicious[name] = indices
print('done')

# benign neurons
print('select benign neurons......', end='')
indices_benign = {}
for name, layer in accumulate_activation_layerwise_benign.items():
    if 'conv' in name.lower():
        sum_activ = torch.sum(layer.squeeze(0),(1,2))
    elif 'fc' in name.lower():
        sum_activ = layer.squeeze(0)
    else:
        print('find an unknown layer:' + name)
        raise Exception('accumulate_activation_layerwise_benign should not contains any layers except conv and fc')
    _, indices = torch.topk(torch.abs(sum_activ), math.ceil(len(sum_activ)*args.topk_ratio), largest = True)
    indices_benign[name] = indices
print('done')

# sum_a = torch.sum(a.squeeze(0),(1,2))  #a (1,32,26,26)
# _, indices0 = torch.topk(torch.abs(sum_a), math.ceil(len(sum_a) * args.topk_ratio), largest=True)
# sum_b = torch.sum(b.squeeze(0),(1,2))  #b (1, 64, 24, 24)
# _, indices1 = torch.topk(torch.abs(sum_b), math.ceil(len(sum_b) * args.topk_ratio), largest=True)
# sum_c = c.squeeze(0)  #c (1,128)
# _, indices2 = torch.topk(torch.abs(sum_c), math.ceil(len(sum_c) * args.topk_ratio), largest=True)
# sum_d = d.squeeze(0)  #d (1,10)
# _, indices3 = torch.topk(torch.abs(sum_d), math.ceil(len(sum_d) * args.topk_ratio), largest=True)
#
# # #benign neurons
# sum_a0 = torch.sum(a0.squeeze(0),(1,2))  #a (1,32,26,26)
# _, indices_0 = torch.topk(torch.abs(sum_a0), math.floor(len(sum_a0) * args.topk_ratio), largest=True)
# sum_b0 = torch.sum(b0.squeeze(0),(1,2))  #b (1, 64, 24, 24)
# _, indices_1 = torch.topk(torch.abs(sum_b0), math.floor(len(sum_b0) * args.topk_ratio), largest=True)
# sum_c0 = c0.squeeze(0)  #c (1,128)
# _, indices_2 = torch.topk(torch.abs(sum_c0), math.floor(len(sum_c0) * args.topk_ratio), largest=True)
# sum_d0 = d0.squeeze(0)  #d (1,10)
# _, indices_3 = torch.topk(torch.abs(sum_d0), math.floor(len(sum_d0) * args.topk_ratio), largest=True)
diff_indices = {}
for key, layer in indices_benign.items():
    s = torch.isin(indices_malicious[key], indices_benign[key]).long()
    idx = torch.nonzero(s -1)
    diff_indices[key] = indices_malicious[key][idx].squeeze(1)


############ Step3: manipulating weights in BCRP ####################
bd_param = {}
benign_param = {}
# detach parameters of each layer to a dict
for name, parameters in backdoor_model.named_parameters():
    bd_param[name] = parameters.detach()
for name, parameters in benign_model.named_parameters():
    benign_param[name] = parameters.detach()

for key,indices in diff_indices.items():
    #if 'fc0' in key.lower() or 'fc1' in key.lower() or 'fc2' in key.lower():
    if 'fc1' in key.lower() or 'fc2' in key.lower():
        benign_param[key + '.weight'][indices] = benign_param[key + '.weight'][indices] + \
                                                 args.alpha * (bd_param[key + '.weight'][indices] - benign_param[key + '.weight'][indices])
        #benign_param[key + '.bias'][indices] = benign_param[key + '.bias'][indices] + \
        #                                         args.alpha * (bd_param[key + '.bias'][indices] - benign_param[key + '.bias'][indices])


############ Step4: using the mask(square) with alpha intensity (test data) ####################
with torch.no_grad():
    for name, param in benign_model.named_parameters():
        if name in benign_param:
            param.copy_(benign_param[name])

#test_backdoor_model(benign_model, test_loader)
util.test_backdoor_model(benign_model, test_loader, args.target_label, args.attack_ratio, args.attack_mode, args.device, strength=100)
