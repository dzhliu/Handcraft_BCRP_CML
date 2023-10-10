import torch.nn as nn
import argparse
from data_poison import *
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
    parser.add_argument('--topk_ratio_coarse', type=float, default=0.1) # top k for bd neuron (num of neurons)
    parser.add_argument('--topk_ratio_fine', type=float, default=0.1) # top k for bd weights
    parser.add_argument('--alpha', type=float, default=10)  #10 work
    parser.add_argument('--model', type=str, default='vgg11')
    parser.add_argument('--benign_model_name', type=str, default="vgg11_cifar10_benign_bn_avgp.pt")
    parser.add_argument('--backdoored_model_name', type=str, default="vgg11_batch128_ep200_cifar10_ratio0.1_strength5_modesig_TrainBD.pt")
    # parser.add_argument('--backdoored_model_name', type=str, default="vgg11_batch128_ep1_cifar10_ratio0.1_strength5_modesig_TrainBD_ep1.pt")
    return parser.parse_args()




def run_instance(poison_data, poison_label, args, train_loader, test_loader, num_channels, topk_ratio_fine_fc2fc1, topk_ratio_fine_fc1fc0, topk_ratio_fine_fc0pool, alpha_fc2fc1, alpha_fc1fc0, alpha_fc0pool, do_conv, fc0conv_type):

    ############ Step2: finding backdoor critical routing path (BCRP) ####################
    activation = {}

    def getActivation(name):
        # the hook signature
        def hook(net, input, output):
            activation[name] = output.detach()
        ####squeeze
        return hook

    benign_model = torch.load('./saved_model/'+args.benign_model_name, map_location=args.device)
    #print('benign model:',end='')
    #util.test_clean_model(benign_model,test_loader,args.device)


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
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) or isinstance(layer, nn.AdaptiveAvgPool2d):
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
        elif isinstance(layer, nn.AdaptiveAvgPool2d):
            current_layer_out_dim = layer.output_size
            accumulate_activation_layerwise_benign[name] = torch.zeros(1, 512, 7, 7).to(args.device)
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

    #print("backdoor model:")
    #util.test_backdoor_model(backdoor_model,test_loader,args.target_label,args.attack_ratio,args.attack_mode,'cpu', 100)
    #print("---------------")

    # poison the dataset
    #print('poisoning the dataset...',end=' ')
    #for batch_idx, (data, label) in enumerate(train_loader):
    #    data, label = find_sig_poison(data, label, target_label=args.target_label, attack_ratio=1.0, strength=255, num_channel=num_channels)
    #    #data, label = sig_poison(data, label, target_label=args.target_label, attack_ratio=1.0, strength=255, num_channel=num_channels)
    #    data = data.to(args.device)
    #    label = label.to(args.device)
    #print('done.')

    data = poison_data
    label = poison_label

    backdoor_model.eval()
    backdoor_model = backdoor_model.to(args.device)

    # register hooks for backdoored model
    hooks_malicious = {}
    for name, layer in backdoor_model.named_modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) or isinstance(layer, nn.AdaptiveAvgPool2d):
            hooks_malicious[name] = layer.register_forward_hook(getActivation(name))

    print('capturing malicious activation values......',end='')
    activation = {} # re-initialize the dict to store activation value
    for i in range(args.batch_size):
        #print(f'data sample {i}-th')
        input_tensor = data[i].unsqueeze(0)
        with torch.no_grad():
            out = backdoor_model(input_tensor)
        for name,_ in accumulate_activation_layerwise_malicious.items():
            accumulate_activation_layerwise_malicious[name] += activation[name].to(args.device)
    print('done')

    # calculate the average (benign)
    for name in accumulate_activation_layerwise_malicious:
        accumulate_activation_layerwise_malicious[name] = accumulate_activation_layerwise_malicious[name]/args.batch_size

    # backdoor neurons
    print('select backdoor neurons......', end='')
    indices_malicious = {}
    for name, layer in accumulate_activation_layerwise_malicious.items():
        if 'conv' in name.lower():
            sum_activ = torch.sum(layer.squeeze(0),(1,2))
        elif 'fc' in name.lower():
            sum_activ = layer.squeeze(0)
        elif 'ap' in name.lower():
            sum_activ = layer.view(layer.size(0),-1) # important: make sure the way we flat the adaptiveAvgPool is the same as that we flat it in model.py
        else:
            print('find an unknown layer:'+name)
            raise Exception('accumulate_activation_layerwise_malicious should not contains any layers except conv and fc')

        _, indices = torch.topk(torch.abs(sum_activ), math.ceil(len(sum_activ)*args.topk_ratio_coarse), largest = True)
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
        elif 'ap' in name.lower():
            sum_activ = layer.view(layer.size(0),-1)  # important: make sure the way we flat the adaptiveAvgPool is the same as that we flat it in model.py
        else:
            print('find an unknown layer:' + name)
            raise Exception('accumulate_activation_layerwise_benign should not contains any layers except conv and fc')
        _, indices = torch.topk(torch.abs(sum_activ), math.ceil(len(sum_activ)*args.topk_ratio_coarse), largest = True)
        indices_benign[name] = indices
    print('done')

    #here we obtain idx of weights which is only contained in indices_malicious but not in indices_benign
    diff_indices = {}
    for key, layer in indices_benign.items():
        if not ('conv' in key.lower()):
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

    #fc2->fc1
    a = accumulate_activation_layerwise_malicious['classifier_layers.FC1'].squeeze(0) #4096
    w = bd_param['classifier_layers.FC2.weight'][diff_indices['classifier_layers.FC2']] #1*4096
    _, idx_fc2 = torch.topk(torch.abs(w * a), math.ceil(topk_ratio_fine_fc2fc1*len(diff_indices['classifier_layers.FC1'])*len(diff_indices['classifier_layers.FC2'])),largest=True) #1*34
    benign_param['classifier_layers.FC2' + '.weight'][args.target_label][idx_fc2] = \
        benign_param['classifier_layers.FC2' + '.weight'][args.target_label][idx_fc2] + \
        alpha_fc2fc1 * (bd_param['classifier_layers.FC2' + '.weight'][args.target_label][idx_fc2] -
                      benign_param['classifier_layers.FC2' + '.weight'][args.target_label][idx_fc2]
        )

    #new fc1->fc0: fc1's bd neurons -> fc0's topk bd neurons
    a = accumulate_activation_layerwise_malicious['classifier_layers.FC0'] # 4096
    w = bd_param['classifier_layers.FC1.weight'][idx_fc2].squeeze(0) # 1*34*4096 -> squeeze: 34*4096
    w_a = w * a #34*4096
    _, idx_fc1 = torch.topk(torch.abs(w_a), math.ceil(topk_ratio_fine_fc1fc0*4096),largest=True) # 34*410
    for i in range(len(idx_fc1)):
        benign_param['classifier_layers.FC1' + '.weight'][idx_fc2[0][i]][idx_fc1[i]] = \
        benign_param['classifier_layers.FC1' + '.weight'][idx_fc2[0][i]][idx_fc1[i]] + alpha_fc1fc0 * (
                    bd_param['classifier_layers.FC1' + '.weight'][idx_fc2[0][i]][idx_fc1[i]] -
                    benign_param['classifier_layers.FC1' + '.weight'][idx_fc2[0][i]][idx_fc1[i]])

    if fc0conv_type == 'unique':
        #fc0->AdaptiveAvgPool2d (the layer after the last conv layer)
        a = accumulate_activation_layerwise_malicious['feature_layers.ap0'] # 1*512*7*7 matrix (altogether 25088 elements)
        a = a.view(a.size(0),-1) # flat the matrix to 1*25088
        idx_fc1_unique = idx_fc1.flatten().unique().unsqueeze(0) #idx_fc1:MAT (x rows and y cols) means the neurons
        w = bd_param['classifier_layers.FC0.weight'][idx_fc1_unique].squeeze(0) # (unique(34*410))*25088 elements
        w_a = w * a
        #_, idx_fc0 = torch.topk(torch.abs(w_a), math.ceil(args.topk_ratio_fine*25088),largest=True)
        _, idx_fc0 = torch.topk(torch.abs(w_a), math.ceil(topk_ratio_fine_fc0pool*25088),largest=True)
        for i in range(len(idx_fc0)):
            benign_param['classifier_layers.FC0' + '.weight'][idx_fc1_unique[0][i]][idx_fc0[i]] = \
                benign_param['classifier_layers.FC0' + '.weight'][idx_fc1_unique[0][i]][idx_fc0[i]] \
                    + alpha_fc0pool *( # + args.alpha*(
                        bd_param['classifier_layers.FC0' + '.weight'][idx_fc1_unique[0][i]][idx_fc0[i]]-
                        benign_param['classifier_layers.FC0' + '.weight'][idx_fc1_unique[0][i]][idx_fc0[i]]

                    )
    elif fc0conv_type == 'all':
        # fc0->AdaptiveAvgPool2d (the layer after the last conv layer)
        # another version (34*400 loop)
        a = accumulate_activation_layerwise_malicious['feature_layers.ap0']  # 1*512*7*7 matrix (altogether 25088 elements)
        a = a.view(a.size(0), -1)  # flat the matrix to 1*25088
        w = bd_param['classifier_layers.FC0.weight'][idx_fc1]  # (34*410)*25088 elements
        w_a = copy.deepcopy(w)
        for w_dim0 in range(0, w.shape[0]):
            for w_dim1 in range(0, w.shape[1]):
                w_a[w_dim0][w_dim1] = w_a[w_dim0][w_dim1] * a
        _, idx_fc0 = torch.topk(torch.abs(w_a), math.ceil(topk_ratio_fine_fc0pool * 25088), largest=True)  # 34*410*2509
        for i in range(idx_fc0.shape[0]):
            for j in range(idx_fc0.shape[1]):
                benign_param['classifier_layers.FC0' + '.weight'][idx_fc1[i][j]][idx_fc0[i][j]] = \
                    benign_param['classifier_layers.FC0' + '.weight'][idx_fc1[i][j]][idx_fc0[i][j]] \
                    + alpha_fc0pool * (
                            bd_param['classifier_layers.FC0' + '.weight'][idx_fc1[i][j]][idx_fc0[i][j]] -
                            benign_param['classifier_layers.FC0' + '.weight'][idx_fc1[i][j]][idx_fc0[i][j]]
                    )

    for key,indices in diff_indices.items():
        #pass
        if do_conv and 'conv' in key.lower():
            benign_param[key+'.weight'][indices] = benign_param[key+'.weight'][indices] + \
                args.alpha * (bd_param[key+'.weight'][indices] - benign_param[key+'.weight'][indices])
        #if 'fc0' in key.lower() or 'fc1' in key.lower() or 'fc2' in key.lower():
        # if 'fc0' in key.lower() or 'fc2' in key.lower():
        #if 'fc2' in key.lower():
        #if 'fc0' in key.lower():
        #     benign_param[key + '.weight'][indices] = benign_param[key + '.weight'][indices] + \
        #                                             args.alpha * (bd_param[key + '.weight'][indices] - benign_param[key + '.weight'][indices])
            #benign_param[key + '.bias'][indices] = benign_param[key + '.bias'][indices] + \
            #                                         args.alpha * (bd_param[key + '.bias'][indices] - benign_param[key + '.bias'][indices])


    ############ Step4: using the mask(square) with alpha intensity (test data) ####################
    with torch.no_grad():
        for name, param in benign_model.named_parameters():
            if name in benign_param:
                param.copy_(benign_param[name])

    #test_backdoor_model(benign_model, test_loader)
    bn_acc, bd_acc = util.test_backdoor_model(benign_model, test_loader, args.target_label, args.attack_ratio, args.attack_mode, args.device, strength=255)
    return bn_acc, bd_acc

if __name__ == "__main__":

    file_handler = open('bd_insertion_accuracy.txt', 'w')


    args = parse_args_attack_activation()
    ####    data loader    #####
    train_loader, test_loader, num_channels = util.data_loader(args.model, args.dataset, args.dataset_path, args.batch_size)

    # poison the dataset
    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = find_sig_poison(data, label, target_label=args.target_label, attack_ratio=1.0, strength=255, num_channel=num_channels)
        # data, label = sig_poison(data, label, target_label=args.target_label, attack_ratio=1.0, strength=255, num_channel=num_channels)
        data = data.to(args.device)
        label = label.to(args.device)


    fc0conv_type_list = {'unique'}
    #topk_ratio_fine_fc2fc1_list = {0.05, 0.1, 0.2}
    topk_ratio_fine_fc2fc1_list = {0.05}
    topk_ratio_fine_fc1fc0_list = {0.05, 0.1, 0.2}
    topk_ratio_fine_fc0pool_list = {0.1, 0.2, 0.3, 0.4}
    #alpha_fc2fc1_list = {6, 8, 10}
    #alpha_fc1fc0_list = {6, 8, 10}
    alpha_fc2fc1_list = {0.3, 0.5, 0.7}
    alpha_fc1fc0_list = {0.3, 0.5, 0.7}
    alpha_fc0pool_list = {8, 10, 12, 14, 16}
    do_conv_list = {False}
    num_exps = len(fc0conv_type_list)*len(topk_ratio_fine_fc2fc1_list)*len(topk_ratio_fine_fc1fc0_list)* \
               len(topk_ratio_fine_fc0pool_list)*len(alpha_fc2fc1_list)*len(alpha_fc1fc0_list)*len(alpha_fc0pool_list)* \
               len(do_conv_list)
    current_exp = 0
    for topk_ratio_fine_fc2fc1 in topk_ratio_fine_fc2fc1_list:
        for topk_ratio_fine_fc1fc0 in topk_ratio_fine_fc1fc0_list:
            for topk_ratio_fine_fc0pool in topk_ratio_fine_fc0pool_list:
                for alpha_fc2fc1 in alpha_fc2fc1_list:
                    for alpha_fc1fc0 in alpha_fc1fc0_list:
                        for alpha_fc0pool in alpha_fc0pool_list:
                            for do_conv in do_conv_list:
                                for fc0conv_type in fc0conv_type_list:
                                    print("current param:")
                                    print(f"param:topk_ratio_fine_fc2fc1-{topk_ratio_fine_fc2fc1:.2f}, topk_ratio_fine_fc1fc0-{topk_ratio_fine_fc1fc0:.2f}, topk_ratio_fine_fc0pool-{topk_ratio_fine_fc0pool:.2f}, alpha_fc2fc1-{alpha_fc2fc1:.2f}, alpha_fc1fc0-{alpha_fc1fc0:.2f}, alpha_fc0pool-{alpha_fc0pool:.2f}, do_conv-{do_conv}, fc0conv_type-{fc0conv_type}")
                                    bn_acc, bd_acc = run_instance(data, label, args, train_loader, test_loader, num_channels, topk_ratio_fine_fc2fc1, topk_ratio_fine_fc1fc0, topk_ratio_fine_fc0pool, alpha_fc2fc1, alpha_fc1fc0, alpha_fc0pool, do_conv, fc0conv_type)
                                    print(f"bn_acc={bn_acc:.4f}, bd_acc={bd_acc:.4f}")
                                    current_exp += 1
                                    file_handler.write(f"progress:" + str(current_exp) + "/" + str(num_exps) + "\n")
                                    file_handler.write(f"param:topk_ratio_fine_fc2fc1-{topk_ratio_fine_fc2fc1:.2f}, topk_ratio_fine_fc1fc0-{topk_ratio_fine_fc1fc0:.2f}, topk_ratio_fine_fc0pool-{topk_ratio_fine_fc0pool:.2f}, alpha_fc2fc1-{alpha_fc2fc1:.2f}, alpha_fc1fc0-{alpha_fc1fc0:.2f}, alpha_fc0pool-{alpha_fc0pool:.2f}, do_conv-{do_conv}, fc0conv_type-{fc0conv_type}")
                                    file_handler.write("\n")
                                    file_handler.write(f"bn_acc={bn_acc:.4f}, bd_acc={bd_acc:.4f}\n")
                                    file_handler.write("=========================================================================\n")
                                    file_handler.flush()
    file_handler.close()