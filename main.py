from model import YOLOv1
import torch

model = YOLOv1(split_size=7, num_boxes=2, num_classes=20)
x = torch.randn((2, 3, 448, 448))   #Giving 2 random images of size 3x448x448
print(model(x).shape)
