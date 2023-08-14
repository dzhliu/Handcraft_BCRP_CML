import torch
from model import *
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torch.nn as nn
import argparse
import wandb
from torch.nn.utils import *

#wandb.login(key = 'dc75cefb6f2dcdb92e9435a6fe80bd396ecc7b49')
wandb.init(project="HBCRP-VGG11test", name="vgg11-batch32-ep200-fmnist", entity="dzhliu")  ####here

########   args ################
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./data')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--device', type=str, default="mps")    # cuda:0
    parser.add_argument('--dataset', type=str, default="fmnist")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--model', type=str, default='vgg11') # CNN, vgg11, vgg13, vgg16, vgg19
    return parser.parse_args()

args=parse_args()

criterion = nn.CrossEntropyLoss()

###### test benign model function ###########

def test_model(model, test_loader):
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
    wandb.log({"benign_accuracy": acc})
    return acc


####data loader        #####
transforms_list = []

if 'vgg' in args.model.lower() and args.dataset == 'fmnist':
    transforms_list.append(transforms.Resize(size=224))

transforms_list.append(transforms.ToTensor())
mnist_transform = transforms.Compose(transforms_list)
train_dataset = datasets.FashionMNIST(root = args.dataset_path, train=True, download=True, transform=mnist_transform)
test_dataset = datasets.FashionMNIST(root = args.dataset_path, train=False, download=True, transform=mnist_transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = args.batch_size, shuffle = True)


#check the number of channels and the number of classes of the current dataset
num_classes = len(train_dataset.classes)
num_channels = 1 if len(train_dataset.train_data.shape) == 3 else train_dataset.train_data.shape[1]

##########    model construction    ###########
if args.model == 'CNN':
    model = ClassicCNN().to(args.device)
elif 'vgg' in args.model.lower():
    if args.model.lower() == 'vgg11':
        model = ClassicVGGx('vgg11', num_classes, num_channels).to(args.device)
    if args.model.lower() == 'vgg13':
        model = ClassicVGGx('vgg13', num_classes, num_channels).to(args.device)
    if args.model.lower() == 'vgg16':
        model = ClassicVGGx('vgg16', num_classes, num_channels).to(args.device)
    if args.model.lower() == 'vgg19':
        model = ClassicVGGx('vgg19', num_classes, num_channels).to(args.device)
else:
    raise Exception(f'the specified model {args.model} does not exist. Please check the model parameter')


#### train benign model ######

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, )  #lr=0.01, momentum=0.9, weight_decay=5e-4

for epoch in range(args.epochs):
    model.train()
    print('current epoch  = {}'.format(epoch))
    for batch_idx, (data, label) in enumerate(train_loader):
        data = data.to(args.device)
        label = label.to(args.device)
        output = model(data)
        optimizer.zero_grad()
        #l2_loss = torch.norm(parameters_to_vector(model.parameters()), p=2)
        #loss = criterion(output, label.view(-1, )) + 0.01*l2_loss
        loss = criterion(output, label.view(-1, ))
        loss.backward()
        optimizer.step()

    print('loss  = {}'.format(loss))
    wandb.log({"loss": loss})
    test_model(model, test_loader)

###### save benign model #########
#torch.save(model, './saved_model/l1_benign_model.pt')
torch.save(model, './saved_model/vgg11_fmnist_benign_model.pt')

print('Train benign model done!')








