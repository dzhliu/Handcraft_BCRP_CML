import matplotlib.pyplot as plt
import numpy
import torchvision
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torch.nn as nn
import argparse
from data_poison import *
from models.vgg import *



########   args ################
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./data')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--device', type=str, default="cpu")    # cuda:0
    parser.add_argument('--dataset', type=str, default="cifar10") #fmnist cifar10
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--target_label', type=int, default=7)
    parser.add_argument('--attack_ratio', type=float, default=0.1)
    parser.add_argument('--attack_mode', type=str, default="square") #square, sig or apple
    parser.add_argument('--model', type=str, default="vgg11") #vgg11 resnet18
    parser.add_argument('--benign_model_name', type=str, default="vgg11_cifar10_benign_bn.pt")
    parser.add_argument('--bd_model_name', type=str, default="vgg11_batch128_ep200_cifar10_ratio0.1_strength1.0_modesquare_TrainBD.pt")
    parser.add_argument('--based_on_benign_model', type=bool, default=False)
    return parser.parse_args()

args=parse_args()

criterion = nn.CrossEntropyLoss()


##############   test backdoor model function ##################
def test_backdoor_model(model, test_loader):
    ########### backdoor accuracy ##############
    total_test_number = 0
    correctly_labeled_samples = 0
    model.eval()
    for batch_idx, (data, label) in enumerate(test_loader):
        if args.attack_mode == 'square':
            data, label = square_poison(data, label, args.target_label, attack_ratio = 1.0, strength=255, num_channel=3)##################TODO: pass
            # num_channel
        elif args.attack_mode == 'sig':
            data, label = sig_poison(data, label, args.target_label, attack_ratio= 1.0, strength=255, num_channel=3) ############TODO: pass
            # num_channel
        else:
            raise Exception(f'unknown attack mode:{args.attack_mode}.')
        data = data.to(device=args.device)
        label = label.to(device=args.device)
        output = model(data)
        total_test_number += len(output)
        _, pred_labels = torch.max(output, 1)
        pred_labels = pred_labels.view(-1)

        correctly_labeled_samples += torch.sum(torch.eq(pred_labels, label)).item()
    model.train()

    acc = correctly_labeled_samples / total_test_number
    bd_acc = acc
    print('backdoor accuracy  = {}'.format(acc))

    ########### benign accuracy ##############
    total_test_number = 0
    correctly_labeled_samples = 0
    model.eval()
    for batch_idx, (data, label) in enumerate(test_loader):
        data = data.to(device=args.device)
        label = label.to(device=args.device)
        output = model(data)
        total_test_number += len(output)
        _, pred_labels = torch.max(output, 1)
        pred_labels = pred_labels.view(-1)

        correctly_labeled_samples += torch.sum(torch.eq(pred_labels, label)).item()
    model.train()
    acc = correctly_labeled_samples / total_test_number
    print('benign accuracy  = {}'.format(acc))
    return bd_acc

def test_backdoor_model_fromTrainAttack(model, test_loader):

    ########### backdoor accuracy ##############
    total_test_number = 0
    correctly_labeled_samples = 0
    model.eval()
    for batch_idx, (data, label) in enumerate(test_loader):
        if args.attack_mode == 'square':
            data, label = square_poison(data, label, args.target_label, attack_ratio = 1.0, strength=255, num_channel=3)##################TODO: pass
            # num_channel
        elif args.attack_mode == 'sig':
            data, label = sig_poison(data, label, args.target_label, attack_ratio= 1.0, strength=255, num_channel=3) ############TODO: pass
            # num_channel
        else:
            raise Exception(f'unknown attack mode:{args.attack_mode}.')
        data = data.to(device=args.device)
        label = label.to(device=args.device)
        output = model(data)
        total_test_number += len(output)
        _, pred_labels = torch.max(output, 1)
        pred_labels = pred_labels.view(-1)

        correctly_labeled_samples += torch.sum(torch.eq(pred_labels, label)).item()
    model.train()

    acc = correctly_labeled_samples / total_test_number
    bd_acc = acc
    print('backdoor accuracy  = {}'.format(acc))

    ########### benign accuracy ##############
    total_test_number = 0
    correctly_labeled_samples = 0
    model.eval()
    for batch_idx, (data, label) in enumerate(test_loader):
        data = data.to(device=args.device)
        label = label.to(device=args.device)
        output = model(data)
        total_test_number += len(output)
        _, pred_labels = torch.max(output, 1)
        pred_labels = pred_labels.view(-1)

        correctly_labeled_samples += torch.sum(torch.eq(pred_labels, label)).item()
    model.train()
    acc = correctly_labeled_samples / total_test_number
    print('benign accuracy  = {}'.format(acc))
    return bd_acc

# attention: comment this function when you put the code base on HPC
# I am not sure if sklearn and tsne has been installed in HPC conda environment
def draw_tsne(data_vec, label_vec):

    from sklearn.manifold import TSNE
    import numpy as np

    project_dim = 2

    x_tsne = TSNE(n_components=project_dim, learning_rate='auto').fit_transform(np.array(data_vec))
    import matplotlib.pyplot as plt

    x_by_category = {}
    y_by_category = {}
    z_by_category = {}
    for i in range(10):
        x_by_category[i] = []
        y_by_category[i] = []
        z_by_category[i] = []
    x_by_category[88] = []
    y_by_category[88] = []
    z_by_category[88] = []

    for idx in range(len(x_tsne)):
        label = label_vec[idx].item()
        #label = label_vec[idx]
        x_by_category[label].append(x_tsne[idx][0])
        y_by_category[label].append(x_tsne[idx][1])
        if project_dim == 3:
            z_by_category[label].append(x_tsne[idx][2])
    # get the center points of each labeled cluster
    centeriods = {}
    for i in range(10):
        x = sum(x_by_category[i])/len(x_by_category[i])
        y = sum(y_by_category[i])/len(y_by_category[i])
        if project_dim == 3:
            z = sum(z_by_category[i])/len(z_by_category[i])
            centeriods[i] = [x,y,z]
        else:
            centeriods[i] = [x,y]

    plt.figure(figsize=(8,8))
    if project_dim == 3:
        ax = plt.axes(projection="3d")
    plt.tight_layout()
    for idx in range(10):
        if project_dim == 3:
            ax.scatter3D(x_by_category[idx],y_by_category[idx],z_by_category[idx],label='class '+str(idx), marker='o')
            ax.scatter3D(centeriods[idx][0], centeriods[idx][1], centeriods[idx][2], label='center ' + str(idx), marker='*', s=100)
        else:
            plt.scatter(x_by_category[idx],y_by_category[idx],label='class '+str(idx), marker='o')
            #plt.scatter(centeriods[idx][0], centeriods[idx][1], label='center '+str(idx), marker='*', s=100)
    plt.scatter(x_by_category[88], y_by_category[88], label='class ' + str(88), marker='*')
    plt.legend()
    plt.show()

    return


def load_dataset():
    ####data loader        #####
    transforms_list = []

    if ('vgg' in args.model.lower() or 'resnet' in args.model.lower()) and args.dataset == 'fmnist':
        transforms_list.append(transforms.Resize(size=32))

    transforms_list.append(transforms.ToTensor())
    mnist_transform = transforms.Compose(transforms_list)
    if args.dataset == 'fmnist':
        train_dataset = datasets.FashionMNIST(root=args.dataset_path, train=True, download=True, transform=mnist_transform)
        test_dataset = datasets.FashionMNIST(root=args.dataset_path, train=False, download=True, transform=mnist_transform)
        num_channels = 1
    elif args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=args.dataset_path, train=True, download=True, transform=mnist_transform)
        test_dataset = datasets.CIFAR10(root=args.dataset_path, train=False, download=True, transform=mnist_transform)
        num_channels = 3
    else:
        raise Exception(f'Error, unknown dataset{args.dataset}')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    # select samples
    idx_label_5 = [i for i, x in enumerate(test_dataset.targets) if x == 5]
    test_dataset_5 = [test_dataset[i] for i in idx_label_5]
    test_loader_5 = torch.utils.data.DataLoader(test_dataset_5, batch_size=128, shuffle=False)

    return train_loader,test_loader, test_loader_5, idx_label_5
def get_hidden_feature_tsne_projection_figure():

    train_loader,test_loader, test_loader_5, idx_label_5 = load_dataset()

    # load model
    if args.based_on_benign_model:
        model = torch.load('./saved_model/' + args.benign_model_name, map_location='cpu')
    else:
        model = torch.load('./saved_model/' + args.bd_model_name, map_location='cpu')

    if (False):
        for batch_idx, (data, label) in enumerate(test_loader_5):
            data, label = square_poison(data, label, target_label=5, attack_ratio=1, strength=255, num_channel=num_channels)
            data = data.to('cpu')
            label = label.to('cpu')
            output = model(data)
            for idx in range(0, len(output)):
                res = torch.topk(output[idx], 1, largest=True)
                print('batch-' + str(batch_idx) + ', sample-' + str(idx) + ', model output=', end='')
                print(res)
            loss = criterion(output, label.view(-1, ))
            print('batch-' + str(batch_idx))

    # obtain hidden feature (i.e., the output of the last conv layer)
    hooks = {}
    activation = {}


    def getActivation(name):
        # the hook signature
        def hook(net, input, output):
            activation[name] = output.detach()

        ####squeeze
        return hook


    for name, layer in model.named_modules():
        # if isinstance(layer, nn.AdaptiveAvgPool2d):
        hooks[name] = layer.register_forward_hook(getActivation(name))
    # capture hidden feature with backdoored data
    hidden_features_bdData = []
    label_bdData = []
    num_samples = 0
    for batch_idx, (data, label) in enumerate(test_loader_5):
        data, label = square_poison(data, label, target_label=5, attack_ratio=1, strength=255, num_channel=num_channels)
        data = data.to("cpu")
        label = label.to("cpu")
        for idx in range(0, len(data)):
            with torch.no_grad():
                out = model(data[idx].unsqueeze(0))
            # hidden_features_bdData.append(activation['feature_layers.ap0'].view(-1).numpy())
            hidden_features_bdData.append(activation['classifier_layers.FC1'].view(-1).numpy())
            label_bdData.append(numpy.array(88))
            num_samples += 1
        if num_samples > 500:
            break
    # capture hidden feature with normal data
    hidden_features_bnData = []
    label_bnData = []
    num_samples = 0
    for batch_idx, (data, label) in enumerate(test_loader):
        for idx in range(0, len(data)):
            with torch.no_grad():
                out = model(data[idx].unsqueeze(0))
            # hidden_features_bnData.append(activation['feature_layers.ap0'].view(-1).numpy())
            hidden_features_bdData.append(activation['classifier_layers.FC1'].view(-1).numpy())
            label_bnData.append(label[idx].numpy())
            num_samples += 1
        if (num_samples > 2000):
            break
    for hook in hooks:
        hooks[hook].remove()

    hidden_features_combine = []
    label_combine = []
    hidden_features_combine += hidden_features_bdData
    hidden_features_combine += hidden_features_bnData
    label_combine += label_bdData
    label_combine += label_bnData

    draw_tsne(hidden_features_combine, label_combine)



def draw_figure(activation,model,name, idx):
    grid = torchvision.utils.make_grid(activation.permute(1, 0, 2, 3))
    img = torchvision.transforms.ToPILImage()(grid)
    img.save('./features/sample' +str(idx) + '_' + model + '_' + name + '.png')

def features_print(model,test_loader, args, poison=False):

    activation = {}
    # define hooks
    def getActivation(name):
        def hook(net, input, output):
            activation[name] = output.detach()
        return hook

    # register hooks to model
    hooks = {}
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            hooks[name] = layer.register_forward_hook(getActivation(name))


    model.eval()
    for batch_idx, (data,label) in enumerate(test_loader):
        if poison:
            data, label = square_poison(data, label, target_label=5, attack_ratio=1, strength=255, num_channel=3) ############adjust num of channels
        data = data.to(device=args.device)
        label = label.to(device=args.device)
        for i in range(args.batch_size):
            input_tensor = data[i].unsqueeze(0)
            input_tensor = input_tensor.to(device=args.device)

            if False:
                img = input_tensor[0]
                img = img.numpy()
                img = np.transpose(img, (1,2,0))
                plt.imshow(img)
                plt.show()

            with torch.no_grad():
                out = model(input_tensor)
            for name, _ in activation.items():
                #operations here...
                draw_figure(activation[name],args.model, name, i)
        break
    model.train()

    # deregister the hooks
    for hook in hooks:
        hooks[hook].remove()

    print('Finish feature printing!')





if __name__ == '__main__':

    train_loader, test_loader, test_loader_5, idx_label_5 = load_dataset()

    if args.based_on_benign_model:
        print('load benign model')
        print('model name: ' + str(args.bd_model_name))
        model = torch.load('./saved_model/' + args.benign_model_name, map_location='cpu')
    else:
        print('load backdoor model')
        print('model name: '+str(args.bd_model_name))
        model = torch.load('./saved_model/' + args.bd_model_name, map_location='cpu')

    #test_backdoor_model(model, test_loader)
    #test_backdoor_model(model, test_loader)
    #test_backdoor_model_fromTrainAttack(model, test_loader)
    features_print(model, test_loader_5, args, poison=True)

    print('done')