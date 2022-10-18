import sys
from PyQt6.QtWidgets import QApplication, QHBoxLayout, QMainWindow, QPushButton, QSpinBox, QVBoxLayout, QWidget, \
    QFileDialog, QLabel, QErrorMessage
from PyQt6.QtGui import QImage, QPixmap
import cv2
import numpy as np


class ImageWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScaledContents(True)

    def hasHeightForWidth(self):
        return self.pixmap() is not None

    def heightForWidth(self, w):
        if self.pixmap():
            try:
                return int(w * (self.pixmap().height() / self.pixmap().width()))
            except ZeroDivisionError:
                return 0


def resize_image(image_data, max_img_width, max_img_height):
    scale_percent = min(max_img_width / image_data.shape[1], max_img_height / image_data.shape[0])
    width = int(image_data.shape[1] * scale_percent)
    height = int(image_data.shape[0] * scale_percent)
    newSize = (width, height)
    image_resized = cv2.resize(image_data, newSize, None, None, None, cv2.INTER_AREA)
    return image_resized


def pixmap_from_cv_image(cv_image):
    height, width, _ = cv_image.shape
    bytesPerLine = 3 * width
    qImg = QImage(cv_image.data, width, height, bytesPerLine, QImage.Format.Format_RGB888).rgbSwapped()
    return QPixmap(qImg)


def get_image_mask(img, blue_shade, diff):
    lower = np.array([max(0, val) for val in [blue_shade[2] - diff, blue_shade[1] - diff, blue_shade[0] - diff]],
                     dtype=np.uint8)
    upper = np.array([min(255, val) for val in [blue_shade[2] + diff, blue_shade[1] + diff, blue_shade[0] + diff]],
                     dtype=np.uint8)
    mask = cv2.inRange(img, lower, upper)
    percent_traced = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
    return cv2.cvtColor(mask, cv2.COLOR_BGR2RGB), percent_traced


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle("Image Tracer")

        main_layout = QVBoxLayout()
        top_bar_layout = QHBoxLayout()
        image_bar_layout = QHBoxLayout()
        self.source_filename = None
        self.source_image_data = None
        self.result_image_data = None
        self.max_img_height = 400
        self.max_img_width = 600

        select_image_button = QPushButton('Select Image')
        process_image_button = QPushButton('Process!')
        select_image_button.clicked.connect(self.choose_source_image)
        process_image_button.clicked.connect(self.process_image)
        for btn in [select_image_button, process_image_button]:
            btn.setFixedHeight(30)
            btn.setFixedWidth(100)
        self.r_select = QSpinBox()
        self.g_select = QSpinBox()
        self.b_select = QSpinBox()
        self.threshold_select = QSpinBox()
        for start_val, prefix, spinbox in zip([30, 1, 72, 152], ['Threshold', 'R', 'G', 'B'],
                                              [self.threshold_select, self.r_select, self.g_select, self.b_select]):
            spinbox.setPrefix(f'{prefix}:')
            spinbox.setMinimum(0)
            spinbox.setMaximum(255)
            spinbox.setValue(start_val)
            spinbox.setFixedWidth(130)

        top_bar_layout.addWidget(select_image_button)
        top_bar_layout.addWidget(self.r_select)
        top_bar_layout.addWidget(self.g_select)
        top_bar_layout.addWidget(self.b_select)
        top_bar_layout.addWidget(self.threshold_select)
        top_bar_layout.addWidget(process_image_button)

        self.source_image = ImageWidget()
        self.result_image = ImageWidget()
        self.source_image.setMaximumSize(self.max_img_width, self.max_img_height)
        self.result_image.setMaximumSize(self.max_img_width, self.max_img_height)

        source_image_layout = QVBoxLayout()
        source_image_layout.addWidget(QLabel("Source image:"))
        source_image_layout.addWidget(self.source_image)

        result_image_layout = QVBoxLayout()
        result_image_layout.addWidget(QLabel("Result image:"))
        result_image_layout.addWidget(self.result_image)

        image_bar_layout.addLayout(source_image_layout)
        image_bar_layout.addLayout(result_image_layout)

        bottom_bar_layout = QHBoxLayout()
        self.save_button = QPushButton('Save as file')
        self.save_button.clicked.connect(self.save_as_file)
        self.save_button.setFixedWidth(300)
        bottom_bar_layout.addWidget(self.save_button)
        self.percent_traced_label = QLabel()
        bottom_bar_layout.addWidget(self.percent_traced_label)

        main_layout.addLayout(top_bar_layout)
        main_layout.addLayout(image_bar_layout)
        main_layout.addLayout(bottom_bar_layout)
        widget = QWidget()
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)

    def choose_source_image(self):
        self.source_filename = QFileDialog.getOpenFileName()[0]
        self.source_image_data = cv2.imread(self.source_filename)
        source_image_resized = resize_image(self.source_image_data, self.max_img_width, self.max_img_height)
        self.source_image.setPixmap(pixmap_from_cv_image(source_image_resized))

    def process_image(self):
        if self.source_image_data is None:
            error_dialog = QErrorMessage()
            error_dialog.showMessage('No image selected')
            error_dialog.exec()
        else:
            self.result_image_data, percent_traced = get_image_mask(self.source_image_data,
                                                                    [self.r_select.value(), self.g_select.value(),
                                                                     self.b_select.value()],
                                                                    self.threshold_select.value())
            self.percent_traced_label.setText(f'Percent of image traced: {(percent_traced * 100):.3f}%')
            result_image_resized = resize_image(self.result_image_data, self.max_img_width, self.max_img_height)
            self.result_image.setPixmap(pixmap_from_cv_image(result_image_resized))

    def save_as_file(self):
        if self.result_image_data is None:
            error_dialog = QErrorMessage()
            error_dialog.showMessage('No image processed')
            error_dialog.exec()
        else:
            filename = QFileDialog.getSaveFileName(self, 'Save File')[0]
            if len(filename) > 0:
                cv2.imwrite(filename, self.result_image_data)


app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()
