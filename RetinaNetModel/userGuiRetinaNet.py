import sys
import json
import os
from typing import Optional, List, Tuple
import torch
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QTextEdit
from PyQt5.QtGui import QPixmap, QImage
from detectron2.config import CfgNode
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
import logging

# Настройка логирования
"""Настройка системы логирования."""
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Определение класса приложения
class ObjectDetectionApp(QWidget):
    """Класс приложения с графическим интерфейсом для детекции объектов с использованием RetinaNet."""

    def __init__(self) -> None:
        """
        Инициализация приложения и настройка модели.
        """
        super().__init__()

        # Загрузка метаданных
        metadata_path: str = "bestRetinaNetModel/metadata.json"
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata_dict: dict = json.load(f)
                thing_classes: List[str] = metadata_dict["thing_classes"]
        else:
            logging.error(f"Файл метаданных {metadata_path} не найден.")
            sys.exit(1)

        # Регистрация датасетов
        data_dir: str = "RetinaNetDataset"
        for split in ["train", "val"]:
            register_coco_instances(
                f"my_dataset_{split}",
                {"thing_classes": thing_classes},
                f"{data_dir}/{split}/result_{split}.json",
                f"{data_dir}/{split}/images"
            )

        # Настройка конфигурации модели
        self.cfg: CfgNode = get_cfg()
        config_path: str = "bestRetinaNetModel/config.yaml"
        try:
            self.cfg.merge_from_file(config_path)
            logging.info("Конфигурация успешно загружена.")
        except FileNotFoundError:
            logging.error(f"Файл конфигурации {config_path} не найден.")
            sys.exit(1)

        # Установка устройства
        weights_path: str = "bestRetinaNetModel/model_final.pth"
        if not torch.cuda.is_available():
            self.cfg.MODEL.DEVICE = "cpu"
            logging.info("Используется CPU.")
        else:
            self.cfg.MODEL.DEVICE = "cuda"
            logging.info("Используется CUDA.")

        if not os.path.exists(weights_path):
            logging.error(f"Веса модели {weights_path} не найдены.")
            sys.exit(1)

        self.cfg.MODEL.WEIGHTS = weights_path
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Порог уверенности для предсказаний

        # Инициализация предиктора
        try:
            self.predictor: DefaultPredictor = DefaultPredictor(self.cfg)
            logging.info("Предиктор успешно инициализирован.")
        except Exception as e:
            logging.error(f"Ошибка инициализации предиктора: {e}")
            sys.exit(1)

        # Загрузка метаданных
        try:
            self.metadata = MetadataCatalog.get("my_dataset_train")
            if not hasattr(self.metadata, 'thing_classes') or not self.metadata.thing_classes:
                logging.error("Метаданные пусты или отсутствует 'thing_classes' после регистрации.")
                sys.exit(1)
            logging.info(f"Метаданные загружены: {self.metadata.thing_classes}")
        except Exception as e:
            logging.error(f"Ошибка загрузки метаданных: {e}")
            sys.exit(1)

        self.initUI()

    def initUI(self) -> None:
        """
        Инициализация графического интерфейса.
        """
        self.setWindowTitle('Приложение для детекции объектов')
        self.setGeometry(100, 100, 600, 500)
        layout: QVBoxLayout = QVBoxLayout()

        # Кнопка для загрузки изображения
        self.upload_button: QPushButton = QPushButton('Загрузить изображение', self)
        self.upload_button.clicked.connect(self.upload_image)
        layout.addWidget(self.upload_button)

        # Метка для отображения изображения
        self.image_label: QLabel = QLabel(self)
        layout.addWidget(self.image_label)

        # Текстовое поле для вывода результатов
        self.detection_info: QTextEdit = QTextEdit(self)
        self.detection_info.setReadOnly(True)
        layout.addWidget(self.detection_info)

        self.setLayout(layout)

    def upload_image(self) -> None:
        """
        Загрузка изображения через диалоговое окно.
        """
        file_name, _ = QFileDialog.getOpenFileName(self, "Открыть изображение", "", "Images (*.png *.xpm *.jpg *.jpeg)")
        if file_name:
            try:
                image: np.ndarray = cv2.imread(file_name)
                if image is None:
                    logging.error(f"Не удалось загрузить изображение {file_name}")
                    return
                logging.info(f"Изображение загружено: {file_name}, размер: {image.shape}")
                self.detect_objects(image)
            except Exception as e:
                logging.error(f"Ошибка в upload_image: {e}")

    def detect_objects(self, image: np.ndarray) -> None:
        """
        Детекция объектов на изображении и отображение результатов.

        Args:
            image (np.ndarray): Входное изображение.
        """
        try:
            # Изменение размера изображения, если оно слишком большое
            max_input_size: int = 1024
            height, width = image.shape[:2]
            if max(height, width) > max_input_size:
                scale: float = max_input_size / max(height, width)
                new_width, new_height = int(width * scale), int(height * scale)
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
                logging.info(f"Изображение изменено до размера: {new_width}x{new_height}")

            # Выполнение предсказания
            outputs: dict = self.predictor(image)
            instances = outputs["instances"].to("cpu")
            pred_classes = instances.pred_classes
            pred_boxes = instances.pred_boxes.tensor.numpy()
            pred_scores = instances.scores.numpy()

            # Фильтрация предсказаний по порогу уверенности
            confidence_threshold: float = 0.7
            valid_indices = pred_scores > confidence_threshold

            output_image: np.ndarray = image.copy()
            detected_objects: List[Tuple[str, float]] = []

            # Аннотация изображения
            for i in range(len(pred_classes)):
                if valid_indices[i]:
                    x1, y1, x2, y2 = map(int, pred_boxes[i])
                    cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    class_id: int = pred_classes[i].item()
                    class_name: str = self.metadata.thing_classes[class_id] if class_id < len(self.metadata.thing_classes) else "Unknown"
                    score: float = pred_scores[i]
                    detected_objects.append((class_name, score))
                    label: str = f"{class_name}: {score:.2f}"
                    cv2.putText(output_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Отображение аннотированного изображения
            self.display_image(output_image)

            # Вывод результатов детекции
            if not detected_objects:
                self.detection_info.setText("❌ Товар не найден на фото")
            else:
                result_text: str = "✅ Найдено: " + ", ".join([f"{name} ({score:.2f})" for name, score in detected_objects])
                self.detection_info.setText(result_text)

        except Exception as e:
            logging.error(f"Ошибка в detect_objects: {e}")

    def display_image(self, image: np.ndarray) -> None:
        """
        Отображение изображения в интерфейсе.

        Args:
            image (np.ndarray): Изображение для отображения.
        """
        try:
            image_rgb: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, channels = image_rgb.shape
            bytes_per_line: int = channels * width
            q_image: QImage = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(q_image).scaled(300, 400))
        except Exception as e:
            logging.error(f"Ошибка в display_image: {e}")


# Основная функция запуска
def main() -> None:
    """
    Запуск приложения с графическим интерфейсом.
    """
    app: QApplication = QApplication(sys.argv)
    ex: ObjectDetectionApp = ObjectDetectionApp()
    ex.show()
    sys.exit(app.exec_())


# Запуск программы
"""Запуск основной функции."""
if __name__ == '__main__':
    main()