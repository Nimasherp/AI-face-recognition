import numpy as np
import torch
import torch.nn as nn
import parameters
import math
import files
from torchvision.ops import box_iou as intersection_over_union


def iou_yolo(bbox1, bbox2):
    # we take the x and y of the intersection of the two boxes
    inter_xmin = torch.max(bbox1[..., 0], bbox2[..., 0])
    inter_ymin = torch.max(bbox1[..., 1], bbox2[..., 1])
    inter_xmax = torch.min(bbox1[..., 2], bbox2[..., 2])
    inter_ymax = torch.min(bbox1[..., 3], bbox2[..., 3]) 

    inter_width = torch.clamp(inter_xmax - inter_xmin, min=0)
    inter_height = torch.clamp(inter_ymax - inter_ymin, min=0)
    intersection_area = inter_width * inter_height

    bbox1_area = (bbox1[..., 2] - bbox1[..., 0]) * (bbox1[..., 3] - bbox1[..., 1])
    bbox2_area = (bbox2[..., 2] - bbox2[..., 0]) * (bbox2[..., 3] - bbox2[..., 1])

    # AUB = A + B - AinterB
    union_area = bbox1_area + bbox2_area - intersection_area

    return intersection_area / union_area


# to make some calculation easier
def xywh_to_xyxy(boxes):
    x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


class Loss(nn.Module):
    def __init__(self,S , **kwargs):
        super(Loss,self).__init__()
        self.coordLoss = nn.MSELoss(reduction='sum')
        self.confLoss = nn.MSELoss(reduction='sum')
        self.grid_y, self.grid_x = torch.meshgrid(torch.arange(S), torch.arange(S), indexing='ij')

        # Weights parameters noobj = no object
        self.lambda_coord = 0.5
        self.lambda_noobj = 5    

    def forward(self, output, expectedOutput):

        S = output.shape[1]
        print(S)
        loss_coord = 0
        loss_conf = 0
        loss_noobj = 0
        for x,expected in zip(output,expectedOutput):

            ioub1 = iou_yolo(xywh_to_xyxy(x[...,1:5]), xywh_to_xyxy(expected[...,1:5]))
            ioub2 = iou_yolo(xywh_to_xyxy(x[...,6:10]), xywh_to_xyxy(expected[...,1:5]))
            ious = torch.stack([ioub1, ioub2], dim=0)
            

            iou_maxes, bestBoxIndice = torch.max(ious,dim=0)
            target_boxes = x
            bbox1 = x[...,1:5]
            bbox2 = x[...,6:10]

            # Only take the right box
            best_mask = ioub1 >= ioub2  # True where first is better, False where second is better
            best_target_coords = torch.where(best_mask.unsqueeze(-1), bbox1, bbox2)  # if true takes first box otherwise second
                  

            
            # is greater or equal to 1, there is a face. This part is just to filer only the grid with faces
            mask = expected[...,0] >= 1
            masked_x = best_target_coords[mask]
            masked_expectedOutput = expected[mask]

            
            bounding_box_coord_expected = masked_expectedOutput[...,1:5]

            # the loss function of YOLO
            loss_coord += self.lambda_coord * self.coordLoss(
                masked_x, 
                bounding_box_coord_expected
            )

            loss_conf += self.confLoss(
                iou_maxes[mask],
                masked_expectedOutput[...,0]
            )

            loss_noobj += self.lambda_noobj * self.confLoss(
                iou_maxes[~mask], 
                expected[...,4][~mask]
            )

        return loss_coord + loss_conf + loss_noobj


# Simple Test
# target = torch.tensor([[[1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 3.0, 3.0, 1.0],
#                        [1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 2.0]],
#                       [[0.5, 0.5, 1.0, 1.0, 1.0, 1.5, 1.5, 2.0, 1.5, 1.5],
#                        [0.5, 1.0, 1.5, 2.0, 2.5, 1.0, 0.5, 1.5, 2.5, 3.0]]])

# output = torch.tensor([[[0.0, 1.0, 1.5, 1.5, 1.0],
#                        [1.0, 2.0, 2.5, 2.5, 2.0]],
#                       [[0.0, 1.5, 2.0, 1.0, 1.0],
#                        [0.0, 1.5, 2.0, 2.5, 1.5]]])
# loss = Loss(5*2)
# lecalcul = loss.forward(target, output)
# print(lecalcul)
