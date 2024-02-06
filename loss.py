import torch
import torch.nn as nn
from utils import intersection_over_union


class YoloLoss(nn.Module):

    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")

        self.S = S
        self.B = B
        self.C = C

        self.lambda_noobj = 0.5
        self.lambda_coord = 5
        
    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)
        # predictions are shaped (BATCH_SIZE, S, S, (C+B*5))
        # target are shaped (BATCH_SIZE, S, S, (C+5))

        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])   #(BATCH_SIZE, S, S, 1)
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])   #(BATCH_SIZE, S, S, 1)
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        iou_maxes, bestbox = torch.max(ious, dim=0) # bestbox --> (BATCH_SIZE, S, S, 1). Each entry gives the box 
        # with the highest IOU with the prediction for that cell. 
        # Note that bestbox will be indices of 0, 1 for which bbox was best

        exists_box = target[..., 20].unsqueeze(3) #Select all the values in the other axices
        #that belongs to the 20th value in last axis 3 which is the target probability to contain an object.
        #(BATCH_SIZE, S, S, 1)
        #in paper this is Iobj_i = 1 if a ground truth box belongs to cell i
        #                        = 0 if a ground truth box does not belong to cell i




        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #

        # Set boxes with no object in them to 0. We only take out one of the two 
        # predictions, which is the one with highest Iou calculated previously.
        box_predictions = exists_box * ((bestbox * predictions[..., 26:30] + (1 - bestbox) * predictions[..., 21:25]))
        #If the highest IOU is in the box 1, then output the 4 cordinates for box 1.
        # box_predictions --> (BATCH_SIZE, S, S, 4) if that cell contains a ground truth box.

        box_targets = exists_box * target[..., 21:25]
        #If a ground truth box exists in that cell then, box_targets --> (BATCH_SIZE, S, S, 4)

        # Take sqrt of width, height of boxes to ensure that
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(torch.abs(box_predictions[..., 2:4] + 1e-6))

        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4]) # --> (BATCH_SIZE, S, S, 1)

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )

        "Does this include the first raw of the loss function where the centroid difference is penalized?"





        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        # pred_box is the confidence score for the bbox with highest IoU
        pred_box = (bestbox * predictions[..., 25:26] + (1 - bestbox) * predictions[..., 20:21])

        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21]),
        )




        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        #Both bounding boxes in the cells with no ground truth boxes are considered
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )

        '''Even when the exists_box = 1 for a specific cell, it contains one bounding box (The one with lower probablitiy)
        that should be included in the lambda_noobj?'''




        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #

        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2,),
            torch.flatten(exists_box * target[..., :20], end_dim=-2,),
        )





        loss = (
            self.lambda_coord * box_loss  # first two rows in paper
            + object_loss  # third row in paper
            + self.lambda_noobj * no_object_loss  # forth row
            + class_loss  # fifth row
        )

        return loss
