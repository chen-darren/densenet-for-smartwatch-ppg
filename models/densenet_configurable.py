# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 11:14:46 2024

@author: dchen
"""

# Implementation inspired by the DenseNet paper:
# "Densely connected convolutional networks" by Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q.
# Published in Proceedings of the IEEE conference on computer vision and pattern recognition, 2017.
# Available at: https://arxiv.org/abs/1608.06993

import math
import torch
import torch.nn as nn


# Basic convolutional block comprised of batch norm, ReLU, and 3x3 conv without bottleneck
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, drop_rate, activation):
        super(BasicBlock, self).__init__() # Initializes the superclass of BasicBlock (nn.Module)
        
        self.bn = nn.BatchNorm2d(in_channels)
        self.activation = activation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, # 3x3 convolution without changing output image dimensions
                              stride=1, padding=1, bias=False)
        self.dropout = nn.Dropout(p=drop_rate)
        self.drop_rate = drop_rate
        
    def forward(self, x):
        out1 = self.bn(x)
        out2 = self.activation(out1)
        out3 = self.conv(out2)
        out = out3
        
        if self.drop_rate > 0: # If dropout is set to greater than 0, perform dropout
            out = self.dropout(out)
            
        return torch.cat([x, out], 1) # Direct connections from any layer to all subsequent layers in the same DenseBlock
                                      # Does so by concatenating the output feature map to the input feature map (ResNet summates these)


# Basic convolutional block (i.e. BasicBlock) with added bottleneck layer
class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, drop_rate, activation): # Initializes the superclass of BottleneckBlock (nn.Module)
        super(BottleneckBlock, self).__init__()
        
        inter_channels = out_channels * 4 # Number of channels after bottleneck
        
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.activation = activation
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, # 1x1 convolution that reduces the number of channels
                               stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, out_channels, kernel_size=3, # 3x3 convolution without changing output image dimensions
                               stride=1, padding=1, bias=False)
        self.dropout = nn.Dropout(p=drop_rate)
        self.drop_rate = drop_rate
        
    def forward(self, x):
        out1 = self.bn1(x)
        out2 = self.activation(out1)
        out3 = self.conv1(out2) # Output of 1x1 conv is not concatenated in order to maintain efficiency and lower number of channels
        out4 = self.bn2(out3)
        out5 = self.activation(out4)
        out6 = self.conv2(out5)
        out = out6
        
        if self.drop_rate > 0: # If dropout is set to greater than 0, perform dropout
            out = self.dropout(out)
            
        return torch.cat([x, out], 1) # Direct connections from any layer to all subsequent layers in the same DenseBlock
                                      # Does so by concatenating the output feature map to the input feature map (ResNet summates these)
                                      

# Transitional block comprised of 1x1 convolution and 2x2 average pooling with stride=2
class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, drop_rate, activation): # Initializes the superclass of TransitionBlock (nn.Module)
        super(TransitionBlock, self).__init__() 
        
        self.bn = nn.BatchNorm2d(in_channels)
        self.activation = activation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, # 1x1 convolution without changing output image dimensions
                              stride=1, padding=0, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=drop_rate)
        self.drop_rate = drop_rate
        
    def forward(self, x):
        out1 = self.bn(x)
        out2 = self.activation(out1)
        out3 = self.conv(out2)
        out4 = self.pool(out3)
        out = out4
        
        if self.drop_rate > 0: # If dropout is set to greater than 0, perform dropout
            out = self.dropout(out)
            
        return out


# Dense block
class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate, block, drop_rate, activation): # Growth rate represents the number of output channels
        super(DenseBlock, self).__init__()
        
        self.layer = self._make_layer(num_layers, in_channels, growth_rate, block, drop_rate, activation)
        
    def _make_layer(self, num_layers, in_channels, growth_rate, block, drop_rate, activation):
        layers = []
        
        for i in range(num_layers): # Number of input channels increases by the growth rate each layer due to dense connectivity
            layers.append(block(in_channels+i*growth_rate, growth_rate, drop_rate, activation))
            
        return nn.Sequential(*layers) # Returns the layers of each Dense block in sequential order
    
    def forward(self, x):
        out = self.layer(x)
        
        return out


class InitialBlock(nn.Module):
    def __init__(self, img_channels, inter_channels, activation):
        super(InitialBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(img_channels, inter_channels, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.activation = activation
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.bn1(out1)
        out3 = self.activation(out2)
        out4 = self.max_pool(out3)
        out = out4
        
        return out
    
# config = { # Config for best model
#     'num_layers_per_dense': 4,
#     'growth_rate': 16,
#     'compression': 0.8,
#     'bottleneck': False,
#     'drop_rate': 0.1,
#     'class_weights': [59743/38082, 59743/13800, 59743/7861], # [59743/(38082+7861), 59743/13800],
#     'learning_rate': 0.0000215443469003188,
#     'lambda_l1': 0.0000774263682681127,
#     'activation': nn.GELU(),
#     }

# DenseNet model
class DenseNet(nn.Module):
    def __init__(self, img_channels, num_layers_per_dense, n_classes, growth_rate, compression, bottleneck, drop_rate, activation): # growth_rate = 'k' in paper
        super(DenseNet, self).__init__()
        
        inter_channels = 2 * growth_rate # Initial convolution layer results in 2k channels
        self.num_layers_per_dense = num_layers_per_dense

        if bottleneck == True:
            block = BottleneckBlock
        else:
            block = BasicBlock
            
        # Intial convolution
        self.conv1 = InitialBlock(img_channels, inter_channels, activation)
        
        # 1st dense and transition blocks
        self.dense1 = DenseBlock(num_layers_per_dense, inter_channels, growth_rate, block, drop_rate, activation)
        inter_channels = inter_channels + num_layers_per_dense * growth_rate
        trans_channels = math.floor(inter_channels*compression)
        self.trans1 = TransitionBlock(inter_channels, trans_channels, drop_rate, activation)
        inter_channels = trans_channels
        
        # 2nd dense and transition blocks
        self.dense2 = DenseBlock(num_layers_per_dense, inter_channels, growth_rate, block, drop_rate, activation)
        inter_channels = inter_channels + num_layers_per_dense * growth_rate
        trans_channels = math.floor(inter_channels*compression)
        self.trans2 = TransitionBlock(inter_channels, trans_channels, drop_rate, activation)
        inter_channels = trans_channels
        
        # 3rd dense and transition blocks
        self.dense3 = DenseBlock(num_layers_per_dense, inter_channels, growth_rate, block, drop_rate, activation)
        inter_channels = inter_channels + num_layers_per_dense * growth_rate
        trans_channels = math.floor(inter_channels*compression)
        self.trans3 = TransitionBlock(inter_channels, trans_channels, drop_rate, activation)
        inter_channels = trans_channels
            
        # Classification layer
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.activation = activation
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(inter_channels, n_classes)
        self.softmax = nn.Softmax(dim=1)
        self.out_channels = inter_channels
        
        # Iterate over all modules in the neural network
        for mod in self.modules():
            # If the module is a 2D convolutional layer
            if isinstance(mod, nn.Conv2d):
                # Compute the number of elements in the convolutional filter
                n = mod.kernel_size[0] * mod.kernel_size[1] * mod.out_channels
                # Initialize the weights of the convolutional layer with a normal distribution
                # Mean is 0, and standard deviation is based on the number of input units
                mod.weight.data.normal_(0, math.sqrt(2. / n))
            
            # If the module is a 2D batch normalization layer
            elif isinstance(mod, nn.BatchNorm2d):
                # Set the weights of the batch normalization to 1
                mod.weight.data.fill_(1)
                # Set the biases of the batch normalization to 0
                mod.bias.data.zero_()
            
            # If the module is a linear layer (fully connected layer)
            elif isinstance(mod, nn.Linear):
                # Set the biases of the linear layer to 0
                mod.bias.data.zero_()
                
                
    def forward(self, x):
        # Initial convolution
        out1 = self.conv1(x)
        
        # 1st dense and transition blocks
        out2 = self.dense1(out1)
        out3 = self.trans1(out2)
        
        # 2nd dense and transition blocks
        out4 = self.dense2(out3)
        out5 = self.trans2(out4)
        
        # 3rd dense and transition blocks
        out6 = self.dense3(out5)
        out7 = self.trans3(out6)
        
        # Classification
        out8 = self.bn2(out7)
        out9 = self.activation(out8)
        out10 = self.avg_pool(out9) # Reduce the output size to 1x1
        out11 = self.flatten(out10) # Flattens to C-dimensional vector
        out12 = self.fc(out11) # Outputs the score for each class (logits)
        out13 = self.softmax(out12) # Converts the scores to probabilities for each class
        out = out13
        
        # Outputs
        logits = out12
        prediction_proba = out
        predictions = torch.argmax(prediction_proba, dim=1) # Returns the class with the highest probability
        
        return logits, predictions, prediction_proba