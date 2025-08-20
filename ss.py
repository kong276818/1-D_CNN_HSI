import sys
import os
import numpy as np
import cv2
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QSlider, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QFrame, QButtonGroup,
    QRadioButton, QScrollArea
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from SviParser import SviParser

class ThresholdingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.class_id = 2  # Default class ID for labeling
        self.class_labels = {
            'BG': 0, 'PET': 1, 'PE': 2, 'PP': 3, 'PS': 4, 'ABS':     5,
            'PAPER': 6, 'CAN': 7, 'PVC': 8, 'PC': 9, 'Noise': 10
        }
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Thresholding')

        self.imageLabel = QLabel(self)
        self.imageLabel.setFrameStyle(QFrame.Box | QFrame.Plain)

        self.fileNameLabel = QLabel(self)
        self.fileNameLabel.setAlignment(Qt.AlignLeft)

        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setMinimum(0)
        self.slider.setMaximum(255)
        self.slider.setValue(30)
        self.slider.valueChanged.connect(self.updateThreshold)
        self.slider.setFixedWidth(600)

        self.thresholdLabel = QLabel(f'Threshold: {self.slider.value()}', self)
        self.thresholdLabel.setFixedWidth(150)

        self.saveButton = QPushButton('Save', self)
        self.saveButton.clicked.connect(self.saveImage)

        self.loadButton = QPushButton('Load File', self)
        self.loadButton.clicked.connect(self.loadFile)

        self.quitButton = QPushButton('Quit', self)
        self.quitButton.clicked.connect(QApplication.instance().quit)

        self.classButtonGroup = QButtonGroup(self)
        hbox_class_buttons = QHBoxLayout()
        button_width = 100
        button_height = 40
        for class_name, class_id in self.class_labels.items():
            button = QRadioButton(class_name, self)
            button.setFixedSize(button_width, button_height)
            button.toggled.connect(lambda checked, cid=class_id: self.setClassId(cid) if checked else None)
            self.classButtonGroup.addButton(button, class_id)
            hbox_class_buttons.addWidget(button)
        hbox_class_buttons.addStretch(1)

        self.saveButton.setFixedSize(button_width, button_height)
        self.loadButton.setFixedSize(button_width, button_height)
        self.quitButton.setFixedSize(button_width, button_height)
                     
        self.scrollArea = QScrollArea(self)
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaWidget = QWidget()
        self.scrollArea.setWidget(self.scrollAreaWidget)

        self.vbox_main = QVBoxLayout(self.scrollAreaWidget)
        self.vbox_main.addWidget(self.fileNameLabel)
        self.vbox_main.addWidget(self.imageLabel)

        hbox_slider = QHBoxLayout()
        hbox_slider.addWidget(self.thresholdLabel)
        hbox_slider.addWidget(self.slider)
        hbox_slider.addStretch(1)

        hbox_buttons = QHBoxLayout()
        hbox_buttons.addWidget(self.saveButton)
        hbox_buttons.addWidget(self.loadButton)
        hbox_buttons.addWidget(self.quitButton)
        hbox_buttons.addStretch(1)

        self.vbox_main.addLayout(hbox_class_buttons)
        self.vbox_main.addLayout(hbox_slider)
        self.vbox_main.addLayout(hbox_buttons)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.scrollArea)
        self.setLayout(main_layout)

        self.setGeometry(100, 100, 1200, 600)

    def setClassId(self, class_id):
        self.class_id = class_id
        print(f'Selected class ID: {self.class_id}')

    def loadFile(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        fileName, _ = QFileDialog.getOpenFileName(
            self, "Open File", "", "Files (*.bin *.npy);;All Files (*)", options=options
        )
        if fileName:
            self.fileName = fileName
            if fileName.lower().endswith('.bin'):
                self.processBinFile()
            elif fileName.lower().endswith('.npy'):
                self.processNpyFile()
            else:
                print("Unsupported file format.")

    def processBinFile(self):
        self.folder_path = os.path.dirname(self.fileName)
        self.name = os.path.splitext(os.path.basename(self.fileName))[0]
        self.fileNameLabel.setText(f'File: {self.name}.bin')

        hsi = SviParser(self.fileName)
        hsi.parse()
        self.image = hsi.images.transpose(2, 0, 1)
        hsi_max = np.max(self.image)
        self.src = (self.image.max(axis=2) / hsi_max * 255).astype(np.uint8)

        png_folder = os.path.join(self.folder_path, '../png')
        if not os.path.exists(png_folder):
            os.makedirs(png_folder)
        cv2.imwrite(os.path.join(png_folder, self.name + '.png'), self.src)

        self.updateImage(self.src)

    def processNpyFile(self):
        self.folder_path = os.path.dirname(self.fileName)
        self.name = os.path.splitext(os.path.basename(self.fileName))[0]
        self.fileNameLabel.setText(f'File: {self.name}.npy')

        # (296, n, 104) 형태 로드
        arr = np.load(self.fileName)  # shape: (296, n, 104)
        if arr.ndim != 3 or arr.shape[0] != 296 or arr.shape[2] != 104:
            print("Unexpected shape:", arr.shape)
            return

        self.image = arr  # 3D 보관
        arr_max = np.max(self.image)
        # 마지막 축(104)에서 최대값 → (296, n) → 0~255 정규화
        self.src = (self.image.max(axis=2) / arr_max * 255).astype(np.uint8)

        png_folder = os.path.join(self.folder_path, '../png')
        if not os.path.exists(png_folder):
            os.makedirs(png_folder)
        cv2.imwrite(os.path.join(png_folder, self.name + '.png'), self.src)

    def updateImage(self, img):
        height, width = img.shape
        qImg = QImage(img.data.tobytes(), width, height, width, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qImg)
        self.imageLabel.setPixmap(pixmap)

    def updateThreshold(self):
        thresh = self.slider.value()
        self.thresholdLabel.setText(f'Threshold: {thresh}')

        if thresh == 0:
            self.binary = np.ones_like(self.src) * 255
        else:
            _, self.binary = cv2.threshold(self.src, thresh, 255, cv2.THRESH_BINARY)

        self.updateImage(self.binary)

    def saveImage(self):
        thresh = self.slider.value()
        png_folder = os.path.join(self.folder_path, '../png')
        savePath = os.path.join(png_folder, self.name + f'_threshold_{thresh}.png')
        cv2.imwrite(savePath, self.binary)
        print(f"Thresholded image saved at {savePath}")

        label = cv2.imread(savePath, cv2.IMREAD_GRAYSCALE)
        train = self.image[label == 255]

        npy_folder = os.path.join(self.folder_path, '../npy_label')
        if not os.path.exists(npy_folder):
            os.makedirs(npy_folder)
        np.save(os.path.join(npy_folder, self.name + '.npy'), train)

        labels = np.full(train.shape[0], self.class_id, dtype=np.int16)
        np.save(os.path.join(npy_folder, self.name + '_labels.npy'), labels)

        print(f'Training data saved with shape: {train.shape}')
        print(f'Labels saved with shape: {labels.shape}')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ThresholdingApp()
    ex.show()
    sys.exit(app.exec_())
