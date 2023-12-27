import torch.nn as nn
from torchvision import models
import torch


class VGG19BNforMNIST_S(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(VGG19BNforMNIST_S, self).__init__()
        # Instantiate the VGG19 model with batch normalization
        self.model = models.vgg19_bn(pretrained=False)
        # Modify the classifier to match the number of classes (MNIST dataset has 10 classes)
        self.model.classifier[6] = torch.nn.Linear(
            in_features=4096, out_features=num_classes
        )

    def forward(self, x):
        return self.model(x)


class VGG19BNforMNIST(nn.Module):
    def __init__(self):
        super(VGG19BNforMNIST, self).__init__()
        # Load a pre-trained VGG19 model with batch normalization
        self.model = models.vgg19_bn(pretrained=True)
        # Modify the classifier to fit MNIST (1 incoming channel, 10 classes)
        self.model.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.model.classifier[6] = nn.Linear(4096, 10)

    def forward(self, x):
        return self.model(x)