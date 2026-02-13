import parameters
from ConvNet import Yolo
import numpy as np
import cv2
import os
import torch
import math
import files

# This will convert the x and y coordinates that are taken from one of the square of the grid,
# to the x and y coordinates according to the whole image like in the label.
def bboxToCoord(bbox, x_idx, y_idx):
    x = (bbox[1] + (x_idx + 1))/(parameters.X_GRID) 
    y = (bbox[2] + (y_idx + 1))/(parameters.Y_GRID)
    return x, y

def detection(model, image):
    output = model(image)
    output = parameters.reshape(output).squeeze(0) # squeeze because of the batch dimensions
    
    if(list(output.shape) != parameters.OUTPUT_SIZE):
        raise ValueError(f"output shape {output.shape} does not match size : {OUTPUT_SIZE}")

    # Separate the the bounding boxes into different arrays 
    # Maybe too slow TODO
    output_reshape = output.view(parameters.X_GRID, 
        parameters.Y_GRID,
        parameters.NB_OF_ANCHORBOXES,
        parameters.BOUNDING_BOXES_PARAMETERS
    )

    confidence = output_reshape[..., 0] # confidence score are in first
    bestConf_idx = torch.argmax(confidence, dim=-1) # not using mask
    print(confidence)
    print(bestConf_idx)

model = Yolo(3, flattenSize = 7*7*1024, fullySize = 4096, outputSize = math.prod(parameters.OUTPUT_SIZE))
dataset = files.ImageData(parameters.TRAININGPATH, parameters.LABELPATH)
image_tensor, label_tensor = dataset[0]

detection(model, image_tensor.unsqueeze(0))


    


