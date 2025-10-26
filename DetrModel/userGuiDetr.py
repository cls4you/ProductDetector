import os
import sys
from typing import Tuple, List, Optional, Dict, Any
import torch
from transformers import DetrForObjectDetection, DetrImageProcessor
import cv2
import supervision as sv
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QTextEdit, QFileDialog, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
import logging

# Настройка логирования и путей
"""Настройка логирования и путей к модели и данным."""
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

HOME: str = os.getcwd()  # Текущая рабочая директория
MODEL_PATH: str = os.path.join(HOME, 'bestDetrModel')  # Путь к сохранённой модели
DEVICE: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # Устройство для вычислений
CONFIDENCE_THRESHOLD: float = 0.7  # Порог уверенности для детекции объектов


# Загрузка модели и процессора
"""Инициализация модели Detr и процессора изображений."""
try:
    logging.info("Загрузка модели и процессора изображений...")
    model: DetrForObjectDetection = DetrForObjectDetection.from_pretrained(MODEL_PATH)
    image_processor: DetrImageProcessor = DetrImageProcessor.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    logging.info("Модель и процессор успешно загружены.")
except Exception as e:
    logging.error(f"Ошибка при загрузке модели или процессора: {e}")
    sys.exit(1)


# Функция обработки изображения
def process_image(image_path: str) -> Tuple[Optional[np.ndarray], List[Any]]:
    """
    Обработка изображения и выполнение детекции объектов с помощью модели Detr.

    Args:
        image_path (str): Путь к изображению.

    Returns:
        Tuple[Optional[np.ndarray], List[Any]]: Аннотированное изображение и список найденных объектов или сообщение об ошибке.
    """
    try:
        logging.info(f"Обработка изображения: {image_path}")
        # Загрузка и конвертация изображения
        image: np.ndarray = cv2.imread(image_path)
        if image is None:
            raise ValueError("Не удалось загрузить изображение с помощью OpenCV.")
        image_rgb: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        logging.info("Изображение загружено и конвертировано в RGB.")

        # Выполнение предсказания моделью
        with torch.no_grad():
            logging.info("Запуск инференса...")
            inputs: Dict[str, torch.Tensor] = image_processor(images=image_rgb, return_tensors='pt').to(DEVICE)
            outputs = model(**inputs)
            logging.info(f"Выходы модели: scores={outputs.logits.softmax(-1)[0, :, :-1].max(-1)[0]}, boxes={outputs.pred_boxes[0]}")
            target_sizes: torch.Tensor = torch.tensor([image_rgb.shape[:2]]).to(DEVICE)
            results: Dict[str, Any] = image_processor.post_process_object_detection(
                outputs=outputs,
                threshold=CONFIDENCE_THRESHOLD,
                target_sizes=target_sizes
            )[0]
            logging.info(f"Обработанные результаты: {results}")

        # Преобразование результатов в формат библиотеки supervision
        detections: sv.Detections = sv.Detections.from_transformers(transformers_results=results)
        logging.info(f"Детекции: {detections}")

        # Получение меток классов
        id2label: Dict[int, str] = model.config.id2label
        logging.info(f"id2label: {id2label}")

        # Формирование списка найденных объектов
        detected_objects: List[Tuple[str, float]] = []
        for confidence, class_id in zip(detections.confidence, detections.class_id):
            class_name: str = id2label.get(int(class_id), "Unknown")
            detected_objects.append((class_name, confidence))
        logging.info(f"Найденные объекты: {detected_objects}")

        # Аннотация изображения
        box_annotator: sv.BoxAnnotator = sv.BoxAnnotator()
        label_annotator: sv.LabelAnnotator = sv.LabelAnnotator()
        labels: List[str] = [f"{class_name} {confidence:.2f}" for class_name, confidence in detected_objects]
        annotated_frame: np.ndarray = box_annotator.annotate(scene=image_rgb.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        logging.info("Изображение успешно аннотировано.")

        return annotated_frame, detected_objects
    except Exception as e:
        logging.error(f"Ошибка в process_image: {e}")
        return None, [f"Error: {str(e)}"]


# Определение графического интерфейса
class MainWindow(QMainWindow):
    """Основное окно приложения с графическим интерфейсом для Detr."""

    def __init__(self) -> None:
        """
        Инициализация главного окна приложения.
        """
        super().__init__()
        self.setWindowTitle("Detr Object Detection with PyQt5")
        self.setGeometry(100, 100, 600, 500)

        # Настройка центрального виджета и layout
        self.central_widget: QWidget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout: QVBoxLayout = QVBoxLayout(self.central_widget)

        # Кнопка для загрузки изображения
        self.load_button: QPushButton = QPushButton("Загрузить изображение")
        self.load_button.clicked.connect(self.load_and_display_image)
        self.layout.addWidget(self.load_button)

        # Метка для отображения изображения
        self.image_label: QLabel = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label)

        # Текстовое поле для вывода результатов
        self.result_text: QTextEdit = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setFixedHeight(100)
        self.layout.addWidget(self.result_text)

    def load_and_display_image(self) -> None:
        """
        Загрузка изображения через диалоговое окно и отображение результатов детекции.
        """
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Выберите изображение", "",
                "Image Files (*.jpg *.jpeg *.png)"
            )
            if file_path:
                logging.info(f"Выбран путь к изображению: {file_path}")
                annotated_image, detected_objects = process_image(file_path)

                # Отображение аннотированного изображения
                if annotated_image is not None:
                    height, width, channel = annotated_image.shape
                    bytes_per_line: int = 3 * width
                    q_image: QImage = QImage(annotated_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
                    pixmap: QPixmap = QPixmap.fromImage(q_image)
                    scaled_pixmap: QPixmap = pixmap.scaled(300, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.image_label.setPixmap(scaled_pixmap)
                    logging.info("Изображение успешно отображено.")
                else:
                    logging.warning("Аннотированное изображение отсутствует, пропуск отображения.")

                # Вывод результатов детекции
                self.result_text.clear()
                if detected_objects and not isinstance(detected_objects[0], str):  # Проверка на валидные детекции
                    result_text: str = "✅ Найдено: " + ", ".join([f"{name} ({conf:.2f})" for name, conf in detected_objects])
                    self.result_text.setText(result_text)
                else:
                    self.result_text.setText("❌ Товар не найден на фото")
        except Exception as e:
            logging.error(f"Ошибка: {e}")
            self.result_text.clear()
            self.result_text.setText(f"Error: {str(e)}")


# Запуск приложения
"""Запуск графического приложения."""
if __name__ == '__main__':
    app: QApplication = QApplication(sys.argv)
    window: MainWindow = MainWindow()
    window.show()
    sys.exit(app.exec_())