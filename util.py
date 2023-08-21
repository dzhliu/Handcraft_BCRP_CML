import argparse
import data_poison
import torch
from torchvision import datasets, transforms
def data_loader(model_name, dataset_name, dataset_path, batch_size):

    transforms_list = []

    if 'vgg' in model_name.lower() and dataset_name == 'fmnist':
        transforms_list.append(transforms.Resize(size=32))

    transforms_list.append(transforms.ToTensor())
    mnist_transform = transforms.Compose(transforms_list)

    if dataset_name == 'fmnist':
        train_dataset = datasets.FashionMNIST(root=dataset_path, train=True, download=True, transform=mnist_transform)
        test_dataset = datasets.FashionMNIST(root=dataset_path, train=False, download=True, transform=mnist_transform)
        num_channels = 1
    elif dataset_name == 'cifar10':
        train_dataset = datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=mnist_transform)
        test_dataset = datasets.CIFAR10(root=dataset_path, train=False, download=True, transform=mnist_transform)
        num_channels = 3
    else:
        raise Exception(f'Error, unknown dataset{dataset_name}')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, num_channels

def test_backdoor_model(model, test_loader, target_label, attack_ratio, trigger_type, device, strength):
    ########### backdoor accuracy ##############
    total_test_number = 0
    correctly_labeled_samples = 0
    model.eval()
    for batch_idx, (data, label) in enumerate(test_loader):
        if trigger_type == 'sig':
            data, label = data_poison.sig_poison(data, label, target_label, attack_ratio = 1.0, strength=strength, num_channel=3)
        elif trigger_type == 'find_sig':
            data, label = data_poison.find_sig_poison(data, label, target_label, attack_ratio=1.0, strength=strength, num_channel=3)
        elif trigger_type == 'square':
            data, label = data_poison.square_poison(data, label, target_label, attack_ratio=1.0, strength=strength, num_channel=3)
        else:
            raise Exception(f'Error, unknown trigger type {trigger_type} in util.py')
        data = data.to(device=device)
        label = label.to(device=device)
        output = model(data)
        total_test_number += len(output)
        _, pred_labels = torch.max(output, 1)
        pred_labels = pred_labels.view(-1)

        correctly_labeled_samples += torch.sum(torch.eq(pred_labels, label)).item()
    model.train()

    acc = correctly_labeled_samples / total_test_number
    print('backdoor accuracy  = {}'.format(acc))
    ########### benign accuracy ##############
    total_test_number = 0
    correctly_labeled_samples = 0
    model.eval()
    for batch_idx, (data, label) in enumerate(test_loader):
        data = data.to(device=device)
        label = label.to(device=device)
        output = model(data)
        total_test_number += len(output)
        _, pred_labels = torch.max(output, 1)
        pred_labels = pred_labels.view(-1)

        correctly_labeled_samples += torch.sum(torch.eq(pred_labels, label)).item()
    model.train()
    acc = correctly_labeled_samples / total_test_number
    print('benign accuracy  = {}'.format(acc))


def test_clean_model(model, test_loader, device):
    ########### benign accuracy ##############
    total_test_number = 0
    correctly_labeled_samples = 0
    model.eval()
    for batch_idx, (data, label) in enumerate(test_loader):
        data = data.to(device=device)
        label = label.to(device=device)
        output = model(data)
        total_test_number += len(output)
        _, pred_labels = torch.max(output, 1)
        pred_labels = pred_labels.view(-1)
        correctly_labeled_samples += torch.sum(torch.eq(pred_labels, label)).item()
    model.train()
    acc = correctly_labeled_samples / total_test_number
    print('benign accuracy  = {}'.format(acc))