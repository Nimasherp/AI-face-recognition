import numpy as np
import torch


TRAININGPATH = "/home/etu/Documents/document_Lenovo/Documents/Programing/Pyhton_Project/facesClassification/FaceData/images/train/"
LABELPATH = "/home/etu/Documents/document_Lenovo/Documents/Programing/Pyhton_Project/facesClassification/FaceData/labels/train/"

# This list uses tuple for convolutional layers
# string fro max pooling
# and list for multiple convolutional layers
ARCHITECTURE_CONFIGURATION = [
    #(size of kernels, numbers of filters, stride, padding)
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),

]

def reshape(flattenOutput):
    return flattenOutput.reshape(-1, X_GRID, Y_GRID, NB_OF_ANCHORBOXES * BOUNDING_BOXES_PARAMETERS)

NB_OF_ANCHORBOXES = 2
X_GRID = 7
Y_GRID= 7
BOUNDING_BOXES_PARAMETERS = 5

BATCH_SIZE = 2

NUM_IMAGES = 160

NUM_EPOCH = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONFIDENCE_TRESHOLD = 0.5

OUTPUT_SIZE = [7, 7, (NB_OF_ANCHORBOXES * BOUNDING_BOXES_PARAMETERS) ]
