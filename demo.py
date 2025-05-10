
import cv2
import torch
from model.yolo_resnet import YOLOResNet
from utils.transforms import get_transforms
import matplotlib.pyplot as plt

def show_image_with_boxes(image, boxes):
    for box in boxes:
        cls, xc, yc, bw, bh = box
        h, w = image.shape[:2]
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

model = YOLOResNet(num_classes=1)
model.load_state_dict(torch.load("yolo_resnet.pth", map_location=torch.device("cpu")))
model.eval()

image_path = "datasets/test.v4i.yolov5pytorch/valid/images/Image1666_jpg.rf.cc758ec5ebd419a7846a50275df0db67.jpg"
image = cv2.imread(image_path)
tensor = get_transforms()(image).unsqueeze(0)

with torch.no_grad():
    pred = model(tensor)[0]

# NOTE: This is just a placeholder demo. You need to apply post-processing to convert raw outputs to boxes.
# Here we just show the image for now.
show_image_with_boxes(image.copy(), [])
