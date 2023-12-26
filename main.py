from PyQt6.QtWidgets import QApplication, QWidget, QFileDialog, QMainWindow
from interface import Ui_MainWindow
from PyQt6.QtGui import QPixmap
from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt
import os
import cv2
from loguru import logger
import json
from GraffitiBoard import GraffitiBoard


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
        self.pushButton.clicked.connect(self.load_image)  
        self.pushButton_2.clicked.connect(self.load_video)  

        # intialize 
        self.load_image_path = ""
        self.load_image_video = ""

    def load_image(self):
        # Open a QFileDialog to select an image
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image File",
            "",  # Start at the current directory or provide a default path
            "Image Files (*.jpeg; *.jpg; *.png; *.bmp; *.gif)"
        )
        if file_name:
            self.load_image_path = file_name
            logger.info(f"Image loaded: {self.load_image_path}")

    def load_video(self):
        # Open a QFileDialog to select a video
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",  # Start at the current directory or provide a default path
            "Video Files (*.mp4; *.avi; *.mov; *.mkv)"
        )
        if file_name:
            self.load_video_path = file_name
            logger.info(f"Video loaded: {self.load_video_path}")


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec())
