import torch
from MNIST.model import VGG19BNforMNIST_S
from torchsummary import summary
from PIL import Image


def show_mnist_structure():
    # Create an instance of the VGG19BNforMNIST model
    model = VGG19BNforMNIST_S()

    # If you have a GPU, you can transfer the model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Print the summary of the model for an input size of 224x224, which is typical for VGG
    # Note: MNIST images are 28x28, but VGG models expect at least 224x224 input,
    # so you would need to resize the images before feeding them to the model.
    summary(model, (3, 32, 32))


def show_loss_and_accuracy(image_path):
    img = Image.open(image_path)
    img.show()