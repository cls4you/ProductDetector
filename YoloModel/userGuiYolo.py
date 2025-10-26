import sys
from typing import List, Tuple
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap
from ultralytics import YOLO

# Загрузка модели
"""Инициализация модели YOLOv8."""
model: YOLO = YOLO("./bestYoloModel/detect/train/weights/best.pt")  # Загрузка обученной модели


# Определение класса приложения
class YOLOApp(QWidget):
    """Класс приложения с графическим интерфейсом для детекции объектов с использованием YOLOv8."""

    def __init__(self) -> None:
        """
        Инициализация приложения.
        """
        super().__init__()
        self.initUI()

    def initUI(self) -> None:
        """
        Инициализация графического интерфейса.
        """
        self.setWindowTitle("Детектор товаров YOLO")
        self.setGeometry(100, 100, 600, 500)

        # Кнопка для загрузки изображения
        self.btnLoad: QPushButton = QPushButton("Загрузить изображение", self)
        self.btnLoad.clicked.connect(self.loadImage)

        # Метка для отображения изображения
        self.imageLabel: QLabel = QLabel(self)

        # Метка для вывода результатов
        self.resultLabel: QLabel = QLabel("Результат:", self)

        # Настройка layout
        layout: QVBoxLayout = QVBoxLayout()
        layout.addWidget(self.btnLoad)
        layout.addWidget(self.imageLabel)
        layout.addWidget(self.resultLabel)
        self.setLayout(layout)

    def loadImage(self) -> None:
        """
        Загрузка изображения через диалоговое окно.
        """
        filePath, _ = QFileDialog.getOpenFileName(self, "Выбрать изображение", "", "Images (*.png *.jpg *.jpeg)")
        if filePath:
            pixmap: QPixmap = QPixmap(filePath)
            self.imageLabel.setPixmap(pixmap.scaled(400, 300))
            self.detectObject(filePath)

    def detectObject(self, imagePath: str) -> None:
        """
        Детекция объектов на изображении и отображение результатов.

        Args:
            imagePath (str): Путь к изображению.
        """
        img: np.ndarray = cv2.imread(imagePath)
        height, width, _ = img.shape
        min_area: float = 0.05 * width * height  # Минимальная площадь бокса (5% от изображения)

        # Выполнение детекции
        results = model(img)  # Запуск модели YOLO
        detected_objects: List[Tuple[str, float]] = []
        has_valid_object: bool = False

        import random
        colors: dict = {}  # Словарь для хранения случайных цветов для классов
        for result in results:
            for box in result.boxes:
                class_id: int = int(box.cls)
                if class_id not in colors:
                    colors[class_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                confidence: float = float(box.conf)  # Уверенность предсказания
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Координаты бокса
                area: float = (x2 - x1) * (y2 - y1)

                # Фильтрация по уверенности и размеру бокса
                if confidence > 0.7 and area > min_area:
                    class_name: str = model.names[class_id]
                    detected_objects.append((class_name, confidence))
                    has_valid_object = True

                    # Аннотация изображения
                    cv2.rectangle(img, (x1, y1), (x2, y2), colors[class_id], 2)
                    cv2.putText(img, f"{class_name} {confidence:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[class_id], 2)

        # Сохранение и отображение аннотированного изображения
        output_path: str = "lastTestImageYoloModel.jpg"
        cv2.imwrite(output_path, img)
        pixmap: QPixmap = QPixmap(output_path)
        self.imageLabel.setPixmap(pixmap.scaled(300, 400))

        # Вывод результатов детекции
        if not has_valid_object:
            self.resultLabel.setText("❌ Товар не найден на фото")
        else:
            result_text: str = "✅ Найдено: " + ", ".join([f"{name} ({conf:.2f})" for name, conf in detected_objects])
            self.resultLabel.setText(result_text)


# Запуск приложения
"""Запуск графического приложения."""
if __name__ == "__main__":
    app: QApplication = QApplication(sys.argv)
    window: YOLOApp = YOLOApp()
    window.show()
    sys.exit(app.exec_())