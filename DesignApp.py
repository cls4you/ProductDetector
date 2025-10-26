import sys
import json
from typing import Dict, Optional, Set, Any
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QFileDialog, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt
from ultralytics import YOLO
from pyzbar.pyzbar import decode
from PIL import Image
import random


# Определение класса приложения
class CombinedApp(QWidget):
    """Класс приложения для детекции товаров (YOLO) и сканирования штрихкодов с графическим интерфейсом."""

    def __init__(self) -> None:
        """
        Инициализация приложения, загрузка модели и базы данных.
        """
        super().__init__()
        self.model: YOLO = YOLO("./YoloModel/bestYoloModel/detect/train/weights/best.pt")  # Загрузка модели YOLO
        self.database: Dict = self.load_database()  # Загрузка базы данных штрихкодов
        self.yolo_result: Optional[str] = None  # Результат детекции YOLO
        self.barcode_result: Optional[str] = None  # Результат сканирования штрихкода
        self.yolo_detected: bool = False  # Флаг успешной детекции YOLO
        self.barcode_scanned: bool = False  # Флаг успешного сканирования штрихкода
        self.initUI()

    def load_database(self) -> Dict:
        """
        Загрузка базы данных штрихкодов из JSON файла.

        Returns:
            Dict: Словарь с данными штрихкодов или пустой словарь в случае ошибки.
        """
        try:
            with open("databaseCodes.json", "r", encoding="utf-8") as file:
                return json.load(file)
        except FileNotFoundError:
            print("Файл databaseCodes.json не найден. Создаётся пустая база.")
            return {}
        except json.JSONDecodeError:
            print("Ошибка чтения файла databaseCodes.json. Используется пустая база.")
            return {}

    def initUI(self) -> None:
        """
        Инициализация графического интерфейса приложения.
        """
        self.setWindowTitle("Детектор товаров и сканер штрихкодов")
        self.setGeometry(100, 100, 1200, 700)

        # Установка стилей для окна
        self.setStyleSheet("""
            QWidget {
                background-color: #F5F7FA;
                font-family: 'Oswald', 'Creepster', 'Poppins', 'Roboto', 'Arial', 'Times New Roman';
            }
            QLabel {
                color: #333333;
            }
        """)

        # Определение шрифтов
        title_font: QFont = QFont("Oswald", 16, QFont.Bold)
        text_font: QFont = QFont("Oswald", 12, QFont.Bold)

        # Секция YOLO
        self.yoloTitle: QLabel = QLabel("Детекция товаров (YOLO)", self)
        self.yoloTitle.setFont(title_font)
        self.yoloTitle.setAlignment(Qt.AlignCenter)

        self.btnLoadYolo: QPushButton = QPushButton("Загрузить фото для детекции", self)
        self.btnLoadYolo.clicked.connect(self.loadImageYolo)
        self.btnLoadYolo.setFixedHeight(40)
        self.btnLoadYolo.setFont(text_font)
        self.btnLoadYolo.setStyleSheet("""
            QPushButton {
                background-color: #4A90E2;
                color: white;
                border-radius: 10px;
                padding: 10px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #357ABD;
            }
            QPushButton:pressed {
                background-color: #2A6399;
            }
        """)

        self.imageLabelYolo: QLabel = QLabel(self)
        self.imageLabelYolo.setAlignment(Qt.AlignCenter)
        self.imageLabelYolo.setFixedSize(500, 500)
        self.imageLabelYolo.setStyleSheet("""
            QLabel {
                background-color: #FFFFFF;
                border: 2px solid #DDDDDD;
                border-radius: 15px;
                padding: 5px;
            }
            QLabel#imageLabelYolo {
                qproperty-alignment: 'AlignCenter';
            }
        """)

        self.resultLabelYolo: QLabel = QLabel("Результат детекции модели", self)
        self.resultLabelYolo.setFont(text_font)
        self.resultLabelYolo.setAlignment(Qt.AlignCenter)
        self.resultLabelYolo.setStyleSheet("color: #666666;")

        # Секция штрихкодов
        self.barcodeTitle: QLabel = QLabel("Сканер штрихкодов", self)
        self.barcodeTitle.setFont(title_font)
        self.barcodeTitle.setAlignment(Qt.AlignCenter)

        self.btnLoadBarcode: QPushButton = QPushButton("Загрузить фото со штрих(qr)-кодом", self)
        self.btnLoadBarcode.clicked.connect(self.loadImageBarcode)
        self.btnLoadBarcode.setFixedHeight(40)
        self.btnLoadBarcode.setFont(text_font)
        self.btnLoadBarcode.setStyleSheet("""
            QPushButton {
                background-color: #4A90E2;
                color: white;
                border-radius: 10px;
                padding: 10px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #357ABD;
            }
            QPushButton:pressed {
                background-color: #2A6399;
            }
        """)

        self.imageLabelBarcode: QLabel = QLabel(self)
        self.imageLabelBarcode.setAlignment(Qt.AlignCenter)
        self.imageLabelBarcode.setFixedSize(500, 500)
        self.imageLabelBarcode.setStyleSheet("""
            QLabel {
                background-color: #FFFFFF;
                border: 2px solid #DDDDDD;
                border-radius: 15px;
                padding: 5px;
            }
            QLabel#imageLabelBarcode {
                qproperty-alignment: 'AlignCenter';
            }
        """)

        self.resultLabelBarcode: QLabel = QLabel("Результат сканирования штрих(qr)-кода", self)
        self.resultLabelBarcode.setFont(text_font)
        self.resultLabelBarcode.setAlignment(Qt.AlignCenter)
        self.resultLabelBarcode.setStyleSheet("color: #666666;")

        # Метка для сравнения результатов
        self.comparisonLabel: QLabel = QLabel("Сравнение результатов:", self)
        self.comparisonLabel.setFont(text_font)
        self.comparisonLabel.setAlignment(Qt.AlignCenter)
        self.comparisonLabel.setStyleSheet("color: #333333; margin: 10px;")

        # Кнопка сброса
        self.btnReset: QPushButton = QPushButton("Сбросить всё", self)
        self.btnReset.clicked.connect(self.resetAll)
        self.btnReset.setFixedHeight(40)
        self.btnReset.setFont(text_font)
        self.btnReset.setStyleSheet("""
            QPushButton {
                background-color: #E74C3C;
                color: white;
                border-radius: 10px;
                padding: 10px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #C0392B;
            }
            QPushButton:pressed {
                background-color: #A93226;
            }
        """)

        # Настройка компоновки
        yoloLayout: QVBoxLayout = QVBoxLayout()
        yoloLayout.addWidget(self.yoloTitle)
        yoloLayout.addWidget(self.btnLoadYolo, alignment=Qt.AlignCenter)
        yoloLayout.addStretch(1)
        yoloLayout.addWidget(self.imageLabelYolo, alignment=Qt.AlignCenter)
        yoloLayout.addWidget(self.resultLabelYolo)
        yoloLayout.addStretch(1)

        barcodeLayout: QVBoxLayout = QVBoxLayout()
        barcodeLayout.addWidget(self.barcodeTitle)
        barcodeLayout.addWidget(self.btnLoadBarcode, alignment=Qt.AlignCenter)
        barcodeLayout.addStretch(1)
        barcodeLayout.addWidget(self.imageLabelBarcode, alignment=Qt.AlignCenter)
        barcodeLayout.addWidget(self.resultLabelBarcode)
        barcodeLayout.addStretch(1)

        mainLayout: QHBoxLayout = QHBoxLayout()
        mainLayout.addStretch(1)
        mainLayout.addLayout(yoloLayout)
        mainLayout.addLayout(barcodeLayout)
        mainLayout.addStretch(1)

        fullLayout: QVBoxLayout = QVBoxLayout()
        fullLayout.addLayout(mainLayout)
        fullLayout.addWidget(self.comparisonLabel)
        fullLayout.addWidget(self.btnReset, alignment=Qt.AlignCenter)
        fullLayout.setContentsMargins(20, 20, 20, 20)
        self.setLayout(fullLayout)

    def loadImageYolo(self) -> None:
        """
        Загрузка изображения для детекции объектов с помощью YOLO.
        """
        filePath, _ = QFileDialog.getOpenFileName(self, "Выбрать изображение", "", "Images (*.png *.jpg *.jpeg)")
        if filePath:
            pixmap: QPixmap = QPixmap(filePath)
            pixmap = pixmap.scaled(490, 490, Qt.KeepAspectRatio)
            self.imageLabelYolo.setPixmap(pixmap)
            self.detectObject(filePath)

    def detectObject(self, imagePath: str) -> None:
        """
        Детекция объектов на изображении с использованием YOLO.

        Args:
            imagePath (str): Путь к изображению.
        """
        img: np.ndarray = cv2.imread(imagePath)
        height, width, _ = img.shape
        min_area: float = 0.05 * width * height  # Минимальная площадь бокса (5%)

        # Словарь для сопоставления английских названий с русскими
        class_mapping: Dict[str, str] = {
            "bottle": "Бутылка воды",
            "cream": "Крем для рук",
            "kefir": "Пачка кефира",
            "ketchup": "Пачка кетчупа",
            "mayonnaise": "Пачка майонеза",
            "milk": "Бутылка молока",
            "milkcream": "Пачка сливок",
            "mustard": "Банка горчицы",
            "ryazhenka": "Пачка ряженки",
            "sourcream": "Упаковка сметаны"
        }

        results = self.model(img)
        detected_classes: Set[str] = set()
        has_valid_object: bool = False
        colors: Dict[int, tuple] = {}

        for result in results:
            for box in result.boxes:
                class_id: int = int(box.cls)
                if class_id not in colors:
                    colors[class_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                confidence: float = float(box.conf)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area: float = (x2 - x1) * (y2 - y1)

                if confidence > 0.7 and area > min_area:
                    class_name: str = self.model.names[class_id]
                    detected_classes.add(class_mapping.get(class_name, class_name))
                    has_valid_object = True
                    cv2.rectangle(img, (x1, y1), (x2, y2), colors[class_id], 8)
                    cv2.putText(img, f"{confidence:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[class_id], 2)

        # Сохранение и отображение аннотированного изображения
        output_path: str = "lastTestImageYoloModelForDesignApp.jpg"
        cv2.imwrite(output_path, img)
        pixmap: QPixmap = QPixmap(output_path)
        pixmap = pixmap.scaled(490, 490, Qt.KeepAspectRatio)
        self.imageLabelYolo.setPixmap(pixmap)

        # Обновление результатов детекции
        if has_valid_object:
            self.yolo_result = f"Товар найден: {', '.join(detected_classes)}"
            self.yolo_detected = True
            self.resultLabelYolo.setText(self.yolo_result)
        else:
            self.yolo_result = "Товар не найден на фото"
            self.yolo_detected = False
            self.resultLabelYolo.setText(self.yolo_result)

        self.compare_results()

    def loadImageBarcode(self) -> None:
        """
        Загрузка изображения для сканирования штрихкода.
        """
        file_name, _ = QFileDialog.getOpenFileName(self, "Выберите изображение", "", "Image Files (*.png *.jpg *.bmp)")
        if file_name:
            pixmap: QPixmap = QPixmap(file_name)
            pixmap = pixmap.scaled(490, 490, Qt.KeepAspectRatio)
            self.imageLabelBarcode.setPixmap(pixmap)
            self.display_and_scan(file_name)

    def display_and_scan(self, file_name: str) -> None:
        """
        Отображение изображения и сканирование штрихкода.

        Args:
            file_name (str): Путь к изображению.
        """
        code_data: Optional[str] = self.scan_code(file_name)
        if code_data:
            info: str = self.database.get(code_data, "Код не найден в базе данных")
            self.barcode_result = f"Код распознан: {info}"
            self.barcode_scanned = True
            self.resultLabelBarcode.setText(self.barcode_result)
        else:
            self.barcode_result = "Код не распознан"
            self.barcode_scanned = False
            self.resultLabelBarcode.setText(self.barcode_result)

        self.compare_results()

    def scan_code(self, image_path: str) -> Optional[str]:
        """
        Сканирование штрихкода или QR-кода на изображении.

        Args:
            image_path (str): Путь к изображению.

        Returns:
            Optional[str]: Данные штрихкода или None, если код не найден.
        """
        try:
            image: Image.Image = Image.open(image_path)
            decoded_objects: list = decode(image)
            if decoded_objects:
                return decoded_objects[0].data.decode("utf-8")
            return None
        except Exception as e:
            print(f"Ошибка: {e}")
            return None

    def compare_results(self) -> None:
        """
        Сравнение результатов детекции YOLO и сканирования штрихкода.
        """
        if self.yolo_detected and self.barcode_scanned:
            if self.yolo_result and self.barcode_result:
                yolo_text: str = self.yolo_result.lower()
                barcode_text: str = self.barcode_result.lower()

                match_found: bool = False
                if "товар найден" in yolo_text and "код распознан" in barcode_text:
                    yolo_items: list = yolo_text.split("товар найден: ")[1].split(", ") if "товар найден: " in yolo_text else []
                    barcode_info: str = barcode_text.split("код распознан: ")[1] if "код распознан: " in barcode_text else ""
                    for item in yolo_items:
                        if item in barcode_info:
                            match_found = True
                            break

                if match_found:
                    self.comparisonLabel.setText("✅ Результаты совпадают: товар и код соответствуют")
                else:
                    self.comparisonLabel.setText("⚠️ Результаты не совпадают: товар и код не соответствуют")
            else:
                self.comparisonLabel.setText("Сравнение результатов: данные отсутствуют")
        else:
            self.comparisonLabel.setText("Сравнение результатов: загрузите данные для обоих разделов")

    def resetAll(self) -> None:
        """
        Сброс всех данных и интерфейса.
        """
        self.imageLabelYolo.clear()
        self.imageLabelBarcode.clear()
        self.resultLabelYolo.setText("Результат детекции модели")
        self.resultLabelBarcode.setText("Результат сканирования штрих(qr)-кода")
        self.comparisonLabel.setText("Сравнение результатов:")
        self.yolo_result = None
        self.barcode_result = None
        self.yolo_detected = False
        self.barcode_scanned = False


# Запуск приложения
"""Запуск графического приложения."""
if __name__ == "__main__":
    app: QApplication = QApplication(sys.argv)
    window: CombinedApp = CombinedApp()
    window.show()
    sys.exit(app.exec_())