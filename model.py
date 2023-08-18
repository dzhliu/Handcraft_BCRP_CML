import torch
import torch.nn as nn
import torch.nn.functional as F
import collections

#This is to add VGG to the current framework
class ClassicVGGx(nn.Module):

    #definition of commonly used VGG structures
    net_arche_cfg = {
        'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M', 'ap', 'FC1', 'FC2', 'FC3'],
        'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M', 'ap', 'FC1', 'FC2', 'FC3'],
        'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M', 'ap', 'FC1', 'FC2', 'FC3'],
        'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M', 'ap', 'FC1', 'FC2', 'FC3']
    }

    def __init__(self, arch_name, num_classes, num_input_channels):

        super(ClassicVGGx, self).__init__()

        self.num_classes = num_classes

        try:
            arch_structure_def = self.net_arche_cfg[arch_name.lower()]
        except KeyError:
            print(f"the specified vgg structure {arch_name} does not exist. Please define the structure in model.py before use")

        feature_layers = collections.OrderedDict() # conv layer and maxpooling layer
        classifier_layers = collections.OrderedDict() # fc layer and relu layer
        conv_layer_seq = 0
        fc_layer_seq = 0
        maxpooling_layer_seq = 0
        relu_layer_seq = 0
        dropout_layer_seq = 0
        batch_norm2d_seq = 0
        ap_layer_seq = 0

        # the var 'num_input_channels' indicates the number of input channels of the original picture
        # e.g., handwriting photo contains single channel picture, so num_input_channels = 1. RGB picture contains 3 channels, num_input_channels = 3
        in_channels = num_input_channels

        for elem in arch_structure_def:
            if elem == 'M':
                feature_layers['M'+str(maxpooling_layer_seq)] = nn.MaxPool2d(kernel_size=2, stride=2)
                maxpooling_layer_seq += 1
            elif elem == "FC1":
                classifier_layers['FC' + str(fc_layer_seq)] = nn.Linear(512 * 7 * 7, 4096) # the minimum input image size is 32*32
                fc_layer_seq += 1
                classifier_layers['ReLu'+str(relu_layer_seq)] = nn.ReLU(inplace=True)
                relu_layer_seq += 1
                classifier_layers['dp'+str(dropout_layer_seq)] = nn.Dropout()
                dropout_layer_seq += 1
            elif elem == "FC2":
                classifier_layers['FC'+str(fc_layer_seq)] = nn.Linear(4096, 4096)
                fc_layer_seq += 1
                classifier_layers['ReLu'+str(relu_layer_seq)] = nn.ReLU(inplace=True)
                relu_layer_seq += 1
                classifier_layers['dp' + str(dropout_layer_seq)] = nn.Dropout()
                dropout_layer_seq += 1
            elif elem == "FC3":
                classifier_layers['FC'+str(fc_layer_seq)] = nn.Linear(4096, self.num_classes)
                fc_layer_seq += 1
            elif elem == 'ap':
                feature_layers['ap'+str(ap_layer_seq)] = nn.AdaptiveAvgPool2d((7,7))
                ap_layer_seq += 1
            else:
                feature_layers['conv'+str(conv_layer_seq)] = nn.Conv2d(in_channels=in_channels, out_channels=elem, kernel_size=3, padding=1)
                conv_layer_seq += 1
                feature_layers['bn'+str(batch_norm2d_seq)] = nn.BatchNorm2d(elem) ################################
                batch_norm2d_seq += 1
                feature_layers['ReLu'+str(relu_layer_seq)] = nn.ReLU(inplace=True)
                relu_layer_seq += 1
                in_channels = elem
            self.feature_layers = nn.Sequential(feature_layers)
            self.classifier_layers = nn.Sequential(classifier_layers)
        return

    def forward(self, x):
        x = self.feature_layers(x)
        #x = torch.flatten(x, start_dim=1)

        x = x.view(x.size(0),-1)
        x = self.classifier_layers(x)
        return x


class ClassicCNN(nn.Module):
    def __init__(self):
        super(ClassicCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.bn1 = torch.nn.GroupNorm(32,32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.bn2 = torch.nn.GroupNorm(32,64)
        self.conv2_drop = nn.Dropout2d(p=0.25)
        self.fc1 = nn.Linear(9216, 128)  ### 9216
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2_drop(F.max_pool2d(self.conv2(x), 2)))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        # print(x.size())
        x = x.view(-1, 9216)
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, p=0.5)
        x = self.fc2(x)
        return x
