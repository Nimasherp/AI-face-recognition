
import cv2
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torch
import parameters


def reshape(image):
    return cv2.resize(image, (448, 448), interpolation=cv2.INTER_AREA)

def label_to_Array(label):
    result = []
    for faces in label :
        values = list(map(float, faces.strip().split(" ")))
        result.append(values)
    return np.array(result)


class ImageData(Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None):
        self.annotations_dir = annotation_dir
        self.image_dir = image_dir
        self.transform = transform

        self.image_filenames = os.listdir(image_dir)
        
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        image_filename = self.image_filenames[idx]
        name = os.path.splitext(image_filename)[0]

        image = cv2.imread(self.image_dir + image_filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # in case we mix up 
        image = reshape(image)
        
        annotation_path = self.annotations_dir + name + ".txt"

        

        with open(annotation_path, 'r') as f:
            label = f.read().splitlines()


        label = label_to_Array(label)
        
        label_to_coord = torch.zeros((parameters.X_GRID, parameters.Y_GRID, 5))
        for face in label :
            x_grid = int(face[1]*7)
            y_grid = int(face[2]*7)
            width = face[3]
            height = face[4]
            label_to_coord[x_grid][y_grid] = torch.tensor([1, face[1]*7 - x_grid, face[2]*7 - y_grid, width, height])

        image = image / 255.0


        return torch.from_numpy(image).permute(2, 0, 1).float(), label_to_coord
        
    
