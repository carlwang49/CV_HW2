from PyQt6.QtWidgets import QApplication, QFileDialog, QMainWindow
from interface import Ui_MainWindow
from PyQt6 import QtWidgets
from loguru import logger
from GraffitiBoard import GraffitiBoard
from backgroundSubtraction import background_subtraction
from opticalFlow import preprocessing, video_tracking

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

        # intialize 
        self.image_path = ""
        self.video_path = ""
        self.initial_point = None

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


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec())
