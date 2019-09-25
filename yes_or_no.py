import torch
import cv2
import os
import torch.nn as nn
from resnet import resnet50
from torchvision import transforms
from PIL import Image
import numpy as np


MODEL=resnet50
FC=2048

model=MODEL()
model.fc=nn.Linear(FC,2)
model=nn.DataParallel(model)

transform=transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.RandomCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])])

checkpoint=torch.load('./model_best_Adam_0.0004.pth.tar',map_location='cpu')
model.load_state_dict(checkpoint['state_dict'])
model.eval()

def istrue(frame):
    frame = frame[..., ::-1]
    frame=Image.fromarray(np.uint8(frame))
    inputs=transform(frame)
    inputs=torch.unsqueeze(inputs,0)
    predictions = model(inputs)
    print(predictions)
    if predictions[0][0]>predictions[0][1]:
        return True
    else:
        return False
