import parameters
import math
from ConvNet import Yolo
import torch
import torch.nn as nn
import numpy as np
from files import ImageData
from Loss import Loss
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"



def main():

    model = Yolo(3, flattenSize = 7*7*1024, fullySize = 512, outputSize = math.prod(parameters.OUTPUT_SIZE)).to(parameters.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = Loss(S = 7)

    dataset_train = ImageData(parameters.TRAININGPATH, parameters.LABELPATH)
    dataload_train = DataLoader(dataset_train,batch_size= parameters.BATCH_SIZE, shuffle=True)
    for epoch in range(parameters.NUM_EPOCH):
        mean_loss = []

        # just a loading animation to keep track of the iteration
        dataloop = tqdm(dataload_train,leave=True, desc=f"Epoch {epoch+1}")

        for i,(image, label) in enumerate(dataloop):
            image = image.to(parameters.DEVICE)
            label = label.to(parameters.DEVICE)
            output = model(image)
            output = parameters.reshape(output)
            
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # computer slow
            if(i == 100):
                break
            
            mean_loss.append(loss.item())
        
        print(f"Mean loss was {sum(mean_loss) / len(mean_loss)}")
        print(f"mean loss type {((mean_loss))}", f" type of len : {(len(mean_loss))}")
    
    return model



main()