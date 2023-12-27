import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 
from torchvision import transforms
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMessageBox
from PyQt6.QtWidgets import QApplication, QFileDialog, QMainWindow
from interface import Ui_MainWindow
from PyQt6 import QtWidgets
from loguru import logger
from PyQt6.QtGui import QPixmap
from GraffitiBoard import GraffitiBoard
from MNIST.model import VGG19BNforMNIST
from backgroundSubtraction import background_subtraction
from opticalFlow import preprocessing, video_tracking
from dimensionReduction import dimension_reduction
from MNIST.vgg19_bn_mnist_demo import show_mnist_structure, show_loss_and_accuracy
from ResNet50.ResNet50_demo import show_images, show_model_structure, show_comparison, show_inference_catdog


class MainApp(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainApp, self).__init__()
        self.setupUi(self)
        self.graffitiBoard = GraffitiBoard()
        self.gridLayout_2.removeWidget(
            self.graphicsView
        )  # remove the graphicsView from the layout
        self.graphicsView.setParent(None)
        self.graffitiBoard.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )

        self.gridLayout_2.addWidget(
            self.graffitiBoard, 0, 1, 1, 1
        )  # add the graffitiBoard to the layout

        # Connect buttons to methods
        self.pushButton_13.clicked.connect(self.graffitiBoard.reset)
        self.pushButton.clicked.connect(self.load_image)  
        self.pushButton_2.clicked.connect(self.load_video)
        self.pushButton_3.clicked.connect(self.run_background_subtraction)
        self.pushButton_8.clicked.connect(self.run_preprocessing)
        self.pushButton_19.clicked.connect(self.run_video_tracking)
        self.pushButton_9.clicked.connect(self.run_dimension_reduction)
        self.pushButton_12.clicked.connect(self.run_inference_result)
        self.pushButton_10.clicked.connect(self.run_show_mnist_structure)
        self.pushButton_11.clicked.connect(self.run_show_loss_and_accuracy)
        self.pushButton_15.clicked.connect(self.run_show_images)
        self.pushButton_16.clicked.connect(self.run_show_model_structure)
        self.pushButton_14.clicked.connect(self.load_picture)
        self.pushButton_18.clicked.connect(self.run_show_inference_catdog)
        self.pushButton_17.clicked.connect(self.run_show_comparison)
        # intialize 
        self.image_path = ""
        self.video_path = ""
        self.initial_point = None
        self.n_components = None
        self.model_path =  './model/vgg19_bn_mnist.pth'
        self.mnist_image_path = './MNIST/loss_and_accuracy.jpg'
        self.inference_loader = None
        self.ResNet_model_with = './model/cat_dog_with_RandoErasing.pth'
        self.ResNet_model_without = './model/cat_dog_with_RandoErasing.pth'
        self.current_image_path = ""

    def load_image(self):
        # Open a QFileDialog to select an image
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image File",
            "",  # Start at the current directory or provide a default path
            "Image Files (*.jpeg; *.jpg; *.png; *.bmp; *.gif)"
        )
        if file_name:
            self.image_path = file_name
            logger.info(f"Image loaded: {self.image_path}")

    def load_video(self):
        # Open a QFileDialog to select a video
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",  # Start at the current directory or provide a default path
            "Video Files (*.mp4; *.avi; *.mov; *.mkv)"
        )
        if file_name:
            self.video_path = file_name
            logger.info(f"Video loaded: {self.video_path}")
    
    def run_background_subtraction(self):
        background_subtraction(self.video_path)
    
    def run_preprocessing(self):
        self.initial_point = preprocessing(self.video_path)
    
    def run_video_tracking(self):
        
        # Check if a point was successfully obtained
        if self.initial_point is not None:
            video_tracking(self.video_path, self.initial_point)
        else:
            print("Could not find the initial point to track.")
    
    def run_dimension_reduction(self):
        self.n_components = dimension_reduction(self.image_path)
        logger.info(f"The minimum number of components with MSE <= 3.0: {self.n_components}")
    
    def run_show_mnist_structure(self):
        show_mnist_structure()
    
    def run_show_loss_and_accuracy(self):
        show_loss_and_accuracy(self.mnist_image_path)

    def run_inference_result(self):
        # Check if the model path is set
        if not hasattr(self, 'model_path') or not self.model_path:
            QMessageBox.information(self, "Error", "Model path not set. Please load the model first.")
            return

        # Initialize the model architecture
        model = VGG19BNforMNIST()  # Replace with your actual model class

        # Load the trained model state dictionary with map_location to ensure it loads on CPU even if it was trained on GPU
        model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
        model.eval()  # Set the model to evaluation mode

        # Get the image from the graffitiBoard
        image = self.graffitiBoard.get_image()

        # Convert image to grayscale if necessary and resize
        preprocess = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((32, 32)),  # Assuming MNIST 28x28 input
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # MNIST mean and std
        ])
        
        # Apply preprocessing to the image
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

        # Ensure the model is on CPU as torch.cuda.is_available() returned False
        model = model.to('cpu')
        input_batch = input_batch.to('cpu')

        # Inference
        with torch.no_grad():
            outputs = model(input_batch)
            _, predicted = torch.max(outputs, 1)
            prediction = predicted.item()

        # Optional: Show probabilities
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        logger.info(f"Prediction: {prediction}")

        # Show the predicted class label
        _, predicted_class = torch.max(outputs, 1)

        # Show the probability distribution
        plt.figure()
        plt.bar(range(10), probabilities.numpy())  # Assuming you have 10 classes
        plt.title("Probability of each class")
        plt.xlabel("Class")
        plt.ylabel("Probability")
        plt.show()
    
    def load_images(self):
        transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
        # Load the dataset
        inference_dataset = datasets.ImageFolder('./dataset/inference_dataset', transform=transform)
        self.inference_loader = DataLoader(inference_dataset, batch_size=32, shuffle=False)
    

    def load_picture(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Load Image', '', 'Image files (*.jpg *.jpeg *.png *.bmp *.gif)')
        if fname:
            pixmap = QPixmap(fname)
            scaled_pixmap = pixmap.scaled(128, 128, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.current_image_path = fname

            # Save the scaled pixmap for later use in inference
            self.current_pixmap = scaled_pixmap

            scene = QtWidgets.QGraphicsScene(self)
            scene.addPixmap(scaled_pixmap)
            self.graphicsView_2.setScene(scene)
            self.graphicsView_2.fitInView(scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self.graphicsView_2.show()

    def run_show_images(self):
        show_images()
    
    def run_show_model_structure(self):
        show_model_structure(self.ResNet_model_with)
    
    def run_show_comparison(self):
        show_comparison()

    def onTextChange(self, text):
        self.word = text

    def run_show_inference_catdog(self):
        prediction = show_inference_catdog(self.ResNet_model_with, self.current_image_path)
        self.label.setText(f'Prediction: {prediction}')
        

if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec())
