import os
import torch
import matplotlib.pyplot as plt
import cv2
from model import NetG

model = torch.load("mnistG_model.pth")

fig = plt.figure()

inputs = torch.randn(50, 128).float()
outputs = model(inputs)    
for i in range(50):
    img = outputs[i].squeeze().mul(255).clamp(0, 255)
    img = img.byte().numpy()
    plt.subplot(5, 10, i+1)
    plt.axis("off")
    plt.imshow(img, cmap="gray", vmin=0, vmax=255)

plt.savefig("output.png")
    
    