import torch
import cv2
import torch.nn as nn
from resnet import resnet18
from torchvision import transforms
from PIL import Image
import numpy as np
from collections import OrderedDict

GPU_MODE = True
GPU_INDEX = 2
MODEL=resnet18
FC=512

mouse_model=MODEL()
mouse_model.fc=nn.Linear(FC,2)

transform=transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.RandomCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])])

mouse_checkpoint = torch.load('./model_best_mouse.pth.tar')

new_mouse_state_dict = OrderedDict()

# 用了nn.DataParallel的模型需要处理才能在cpu上使用
for k, v in mouse_checkpoint['state_dict'].items():
    name = k[7:]  # remove module.
    new_mouse_state_dict[name] = v

mouse_model.load_state_dict(new_mouse_state_dict)

mouse_model.eval()

if GPU_MODE:
    mouse_model = mouse_model.cuda(GPU_INDEX)
    print("GPU Mode!    GPU INDEX:%s"%str(GPU_INDEX))
else:
    mouse_model = mouse_model.cpu()
    print("Not GPU Mode!")

def ismouse(frame):
    frame = frame[..., ::-1]
    frame=Image.fromarray(np.uint8(frame))
    frame = transform(frame)
    inputs=torch.unsqueeze(frame,0)
    if GPU_MODE:
        inputs = inputs.cuda(GPU_INDEX)
    else:
        inputs = inputs.cpu()
        print(inputs.shape)
    print('*'*50)
    predictions = mouse_model(inputs)
    print(predictions)
    if predictions[0][0]>predictions[0][1]:
        return True
    else:
        return False
