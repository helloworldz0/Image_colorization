from aimodule import Aimodel as Mod
import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import numpy as np
import cv2
from PIL import Image as PILImage , ImageQt

class ColorizationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CNN Image Colorization")
        self.showFullScreen()

        # Image display labels
        self.input_label = QLabel("Grayscale Camera Input")
        self.output_label = QLabel("Colorized Output")

        for label in (self.input_label, self.output_label):
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("border: 2px solid #555; background-color: #222; color: white; font-size: 18px;")

        # Buttons
        self.load_btn = QPushButton("Take Black & White Image")
        self.colorize_btn = QPushButton("Run CNN Colorization")

        self.load_btn.setStyleSheet("font-size: 18px; padding: 10px;")
        self.colorize_btn.setStyleSheet("font-size: 18px; padding: 10px;")

        # Layouts
        img_layout = QHBoxLayout()
        img_layout.addWidget(self.input_label)
        img_layout.addWidget(self.output_label)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.load_btn)
        btn_layout.addWidget(self.colorize_btn)

        main_layout = QVBoxLayout()
        main_layout.addLayout(img_layout)
        main_layout.addLayout(btn_layout)

        self.setLayout(main_layout)

        self.load_btn.clicked.connect(self.load_image)
        self.colorize_btn.clicked.connect(self.colorize_image)

        self.model = None
    def load_image(self):
            self.model=Mod('./tests.keras')
            cam=cv2.VideoCapture(0)
            ret,frame=cam.read()
            cam.release()
            img = 'temp.jpg'
            cv2.imwrite(img,frame)
            self.model.bw(img)
            img = PILImage.fromarray(self.model.gray_img)
            q_img = QImage(self.mode.gray_img.data, 640,480,640, QImage.Format_Grayscale8)

            pixmap = QPixmap(q_img)
            label=self.input_label
            label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def colorize_image(self):
        colorized=self.model.predict()

        img = PILImage.fromarray(colorized)
        q_img = ImageQt.ImageQt(img)
        label=self.output_label
        label.setPixmap(QPixmap(q_img).scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ColorizationApp()
    window.show()
    sys.exit(app.exec_())
