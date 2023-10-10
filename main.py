from models.model import *
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torch.nn as nn
import argparse
import wandb


########   args ################
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./data')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--device', type=str, default="cuda:0")    # cuda:0
    parser.add_argument('--dataset', type=str, default="cifar10") #fmnist cifar10
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--model', type=str, default='vgg11') # CNN, vgg11, vgg13, vgg16, vgg19, resnet18, resnet34
    return parser.parse_args()

args=parse_args()

wandb_name = args.model+"_batch"+str(args.batch_size)+"_ep"+str(args.epochs)+"_"+args.dataset+"_benign"
#wandb.login(key = 'dc75cefb6f2dcdb92e9435a6fe80bd396ecc7b49')
#wandb.init(project="HBCRP-VGG11", name=wandb_name, entity="dzhliu")  ####here
wandb.init(project="HBCRP-resnet11", name=wandb_name, entity="dzhliu")  ####here

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
if ('vgg' in args.model.lower() or 'resnet' in args.model.lower()) and args.dataset.lower() == 'fmnist':
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


#check the number of channels and the number of classes of the current dataset
num_classes = len(train_dataset.classes)


##########    model construction    ###########
if args.model == 'CNN':
    model = ClassicCNN().to(args.device)
elif 'vgg' in args.model.lower():
    model = ClassicVGGx(args.model.lower(), num_classes, num_channels).to(args.device)
elif 'resnet18' in args.model.lower():
    model = ResNet18().to(args.device)
else:
    print('current model:'+args.model)
    raise Exception(f'the specified model {args.model} does not exist. Please check the model parameter')


#### train benign model ######
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  #lr=0.01, momentum=0.9, weight_decay=5e-4

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
    if(epoch % 5 == 0): # save the model every 5 generations (overwrite)
        ###### save benign model #########
        #torch.save(model, './saved_model/l1_benign_model.pt')
        #torch.save(model, './saved_model/vgg11_fmnist_benign_bn_avgp.pt')
        #torch.save(model, './saved_model/resnet_cifar10_benign.pt')
        torch.save(model, './saved_model/vgg11_cifar10_benign_bn.pt')

print('Train benign model done!')








