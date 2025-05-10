
from torchvision import transforms

def get_transforms():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((416, 416)),
        transforms.ToTensor()
    ])
