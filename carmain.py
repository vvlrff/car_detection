import sys
import cv2
import pytesseract
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QFileDialog, QTextEdit
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Распознавание номеров автомобилей')
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        self.canvas = FigureCanvas(Figure())
        self.ax = self.canvas.figure.subplots()
        self.ax.axis('off')  
        self.canvas.hide()

        self.text_edit = QTextEdit(self)
        self.text_edit.setReadOnly(True)

        btn_open = QPushButton('Открыть изображение', self)
        btn_open.clicked.connect(self.open_image)

        layout.addWidget(btn_open)
        layout.addWidget(self.canvas)
        layout.addWidget(self.text_edit)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def open_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Открыть изображение', '', 'Image files (*.jpg *.jpeg *.png)')
        if file_name:
            carplate_img_rgb = cv2.imread(file_name)
            carplate_img_rgb = cv2.cvtColor(carplate_img_rgb, cv2.COLOR_BGR2RGB)
            self.process_image(carplate_img_rgb)

    def process_image(self, carplate_img_rgb):
        carplate_haar_cascade = cv2.CascadeClassifier('haar_cascades/haarcascade_russian_plate_number.xml')
        carplate_rects = carplate_haar_cascade.detectMultiScale(carplate_img_rgb, scaleFactor=1.1, minNeighbors=5)

        if len(carplate_rects) > 0:
            x, y, w, h = carplate_rects[0]
            carplate_img = carplate_img_rgb[y+15:y+h-10, x+15:x+w-20]
            carplate_img_gray = cv2.cvtColor(carplate_img, cv2.COLOR_RGB2GRAY)
            
            number_plate_text = pytesseract.image_to_string(
                carplate_img_gray,
                config='--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            )
            
            with open('plate.txt', 'r') as file:
                plate_numbers = file.read().splitlines()
            
            if number_plate_text.strip() in plate_numbers:
                plate_status = 'Пропуск действителен'
            else:
                plate_status = 'Пропуск недействителен, обратитесь в службу ЗГТ'

            self.text_edit.clear()
            self.text_edit.append('Номер авто: ' + number_plate_text.strip())
            self.text_edit.append(plate_status)

            self.ax.clear()
            self.ax.imshow(carplate_img_gray, cmap='gray')
            self.ax.axis('off')
            self.canvas.draw()
            self.canvas.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainWindow()
    ex.show()
    sys.exit(app.exec_())
