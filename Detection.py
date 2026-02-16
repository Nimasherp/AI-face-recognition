import parameters
from ConvNet import Yolo
import numpy as np
import cv2
import os
import torch
import math
import files


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
    bestConf_idx = torch.argmax(confidence, dim=-1) 

    best_boxes = torch.zeros(
        parameters.X_GRID,
        parameters.Y_GRID,
        parameters.BOUNDING_BOXES_PARAMETERS
    )

    # Now let's loop around bestConf_idx and take the best boxes

    for i in range(parameters.X_GRID):
        for j in range(parameters.Y_GRID):
            best_anchor = bestConf_idx[i, j].item()
            best_boxes[i, j] = output_reshape[i, j, best_anchor]

    # This will convert the x and y coordinates that are taken from one of the square of the grid,
    # to the x and y coordinates according to the whole image like in the label.


    grid_x = torch.arange(parameters.X_GRID).unsqueeze(1)
    grid_y = torch.arange(parameters.Y_GRID)

    conf = best_boxes[..., 0]
    x = best_boxes[..., 1]
    y = best_boxes[..., 2]
    w = best_boxes[..., 3]
    h = best_boxes[..., 4]

    x_global = (x + grid_x)/parameters.X_GRID
    y_global = (y + grid_y)/parameters.Y_GRID

    final_boxes = torch.stack([conf, x_global, y_global, w, h], dim=-1)


    # Now let's remove the boxes that have a confidence score lower than a treshold that we choose; 50%

    conf = final_boxes[..., 0]
    mask = conf>parameters.CONFIDENCE_TRESHOLD

    filtered_boxes = final_boxes[mask]

    print(filtered_boxes)

model = Yolo(3, flattenSize = 7*7*1024, fullySize = 4096, outputSize = math.prod(parameters.OUTPUT_SIZE))
dataset = files.ImageData(parameters.TRAININGPATH, parameters.LABELPATH)
image_tensor, label_tensor = dataset[80]


detection(model, image_tensor.unsqueeze(0))


    


