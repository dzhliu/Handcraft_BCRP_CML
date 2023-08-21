import torch
from model import *
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torch.nn as nn
import argparse
from data_poison import *
import wandb

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
    parser.add_argument('--attack_mode', type=str, default="sig") #square or sig
    parser.add_argument('--model', type=str, default="vgg11") #vgg11 resnet18
    parser.add_argument('--benign_model_name', type=str, default="vgg11_cifar10_benign_bn_avgp.pt")
    return parser.parse_args()

args=parse_args()
wandb_name = args.model+"_batch"+str(args.batch_size)+"_ep"+str(args.epochs)+"_"+args.dataset+"_ratio"+str(
    args.attack_ratio)+"_strength"+str(5)+"_mode"+args.attack_mode
save_model_name = wandb_name+"_TrainBD_ep1.pt"
#wandb.login(key = 'dc75cefb6f2dcdb92e9435a6fe80bd396ecc7b49')
#wandb.init(project="HBCRP-VGG11trainBD", name=wandb_name, entity="dzhliu")  ####here
wandb.init(project="HBCRP-ResnetTrainBD", name=wandb_name, entity="dzhliu")  ####here

criterion = nn.CrossEntropyLoss()

##############   test backdoor model function ##################
def test_backdoor_model(model, test_loader):
    ########### backdoor accuracy ##############
    total_test_number = 0
    correctly_labeled_samples = 0
    model.eval()
    for batch_idx, (data, label) in enumerate(test_loader):
        if args.attack_mode == 'square':
            data, label = square_poison(data, label, args.target_label, attack_ratio = 1.0, strength=5, num_channel=3)##################TODO: pass
            # num_channel
        elif args.attack_mode == 'sig':
            data, label = sig_poison(data, label, args.target_label, attack_ratio= 1.0, strength=5, num_channel=3) ############TODO: pass
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
    wandb.log({"backdoor accuracy": acc})
    ########### backdoor accuracy ##############
    total_test_number = 0
    correctly_labeled_samples = 0
    model.eval()
    for batch_idx, (data, label) in enumerate(test_loader):
        if args.attack_mode == 'square':
            data, label = square_poison(data, label, args.target_label, attack_ratio=1.0, strength=255, num_channel=3)  ##################TODO: pass
            # num_channel
        elif args.attack_mode == 'sig':
            data, label = sig_poison(data, label, args.target_label, attack_ratio=1.0, strength=255, num_channel=3)  ############TODO: pass
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
    str_bd_acc = acc
    print('strong backdoor accuracy  = {}'.format(str_bd_acc))
    wandb.log({"backdoor accuracy":acc, "strong backdoor accuracy":str_bd_acc})
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
    wandb.log({"benign accuracy": acc})
    return bd_acc

############ load benign model ###########################
# model = ClassicCNN().to(args.device)
model = torch.load('./saved_model/'+args.benign_model_name, map_location=args.device)

####data loader        #####
transforms_list = []

if ('vgg' in args.model.lower() or 'resnet' in args.model.lower()) and args.dataset == 'fmnist':
    transforms_list.append(transforms.Resize(size=32))

transforms_list.append(transforms.ToTensor())
mnist_transform = transforms.Compose(transforms_list)
if args.dataset == 'fmnist':
    train_dataset = datasets.FashionMNIST(root = args.dataset_path, train=True, download=True, transform=mnist_transform)
    test_dataset = datasets.FashionMNIST(root = args.dataset_path, train=False, download=True, transform=mnist_transform)
    num_channels = 1
elif args.dataset == 'cifar10':
    train_dataset = datasets.CIFAR10(root = args.dataset_path, train=True, download=True, transform=mnist_transform)
    test_dataset = datasets.CIFAR10(root=args.dataset_path, train=False, download=True, transform=mnist_transform)
    num_channels = 3
else:
    raise Exception(f'Error, unknown dataset{args.dataset}')

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = args.batch_size, shuffle = True)

############ Step1: pre-train backdoor model ####################
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, )  #lr=0.01, momentum=0.9, weight_decay=5e-4
bd_acc = 0

for epoch in range(args.epochs):
    model.train()
    print('current epoch  = {}'.format(epoch))
    for batch_idx, (data, label) in enumerate(train_loader):
        optimizer.zero_grad()

        if args.attack_mode == 'square':
            data, label = square_poison(data, label, target_label=args.target_label, attack_ratio=args.attack_ratio, strength=10, num_channel=num_channels)
        elif args.attack_mode == 'sig':
            data, label = sig_poison(data, label, target_label=args.target_label, attack_ratio=args.attack_ratio, strength=5, num_channel=num_channels)
        else:
            raise Exception(f'unknown attack mode:{args.attack_mode}.')

        data = data.to(args.device)
        label = label.to(args.device)
        output = model(data)
        loss = criterion(output, label.view(-1, ))
        loss.backward()
        optimizer.step()

    print('loss  = {}'.format(loss))
    wandb.log({"loss": loss})

    bd_acc = test_backdoor_model(model, test_loader)
    #if epoch % 5 == 0:
        ###### save backdoor model #########
    torch.save(model, './saved_model/'+save_model_name)

print('Train backdoor model done!')

