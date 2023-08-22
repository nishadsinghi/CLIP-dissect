from torchvision import transforms
import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMG_SIZE = 224

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float),
])

imagenet_normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)