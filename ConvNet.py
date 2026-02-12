import numpy as np
import torch
import torch.nn as nn
import parameters
import math
import files

# here we'll define class that will allow us to defines the convolutional layers 

class ConvNetParts(nn.Module):
    def __init__(self, inputChannels, outputChannels, **kwargs):
        # super method will allow us to integrate proprely pytorch modules 
        super(ConvNetParts, self).__init__()
        # Let's construct its filters
        self.convolution = nn.Conv2d(inputChannels, outputChannels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(outputChannels)
        # Here we'll use a leaky reLu fonction so that we won't have neurons that die, so no 0.
        self.leakyrelu = nn.LeakyReLU(0.1)

    # We'll use all this for the foward propagation
    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.convolution(x)))


class Yolo(nn.Module):
    def __init__(self, inputChannels = 3, **kwargs):
        super(Yolo, self).__init__()
        self.architecture = parameters.ARCHITECTURE_CONFIGURATION
        self.inputChannels = inputChannels
        self.darknet = nn.Sequential(*self.createConvolutionalLayers(self.architecture))
        self.fcs = self.createFcs(**kwargs)
    
    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))

    def createFcs(self, flattenSize, fullySize, outputSize):
        layers = [nn.Linear(flattenSize, fullySize),
                    nn.Dropout(0.0),
                    nn.LeakyReLU(0.1),
                    nn.Linear(fullySize, outputSize)]
        return nn.Sequential(*layers)
    
    def createConvolutionalLayers(self, architecture):
        layers = []
        inputChannels = self.inputChannels
        for blocks in architecture:
            if type(blocks) == tuple:
                layers += [
                    ConvNetParts(inputChannels, blocks[1], kernel_size = blocks[0], stride = blocks[2], padding = blocks[3])
                ]
                inputChannels = blocks[1]
            elif type(blocks) == str:
                layers += [
                    nn.MaxPool2d(kernel_size =2, stride =2)
                ]
            elif type(blocks) == list:
                filter1 = blocks[0]
                filter2 = blocks[1]
                for _ in range(blocks[-1]):
                    layers += [ConvNetParts(inputChannels, filter1[1], kernel_size = filter1[0], stride =filter1[2], padding =filter1[3])]
                    layers += [ConvNetParts(filter1[1], filter2[1],kernel_size =  filter2[0],stride = filter2[2], padding =filter2[3])]
                inputChannels = filter2[1]
            else : 
                raise ValueError("Error in architecture configuration")
        return layers
            

# let's test it with a random batch 
# dataset = files.ImageData(parameters.TRAININGPATH, parameters.LABELPATH)
# image_tensor, label_tensor = dataset[0]

# image_tensor= image_tensor.unsqueeze(0) # Because the Network expect a batchsize, so let's make it 1 for now

# flattensize = 7*7*1024
# yolo = Yolo(3, flattenSize = flattensize, fullySize = 4096, outputSize = math.prod(parameters.OUTPUT_SIZE))
# output = parameters.reshape(Yolo.forward(yolo, image_tensor))
# print(output.shape) # expect 7 * 7 * 10 and times 1 for the batch again

