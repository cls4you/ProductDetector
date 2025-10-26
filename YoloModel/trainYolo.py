import torch
from ultralytics import YOLO
from typing import NoneType


# Функция обучения модели YOLO
def train_yolo() -> NoneType:
    """
    Функция для обучения модели YOLOv8 и экспорта в формат ONNX.
    """
    # Инициализация модели
    model: YOLO = YOLO("yolov8s.pt")  # Загрузка предобученной модели YOLOv8s

    # Запуск обучения
    model.train(
        data="./YoloDataset/augmented.yaml",  # Путь к файлу конфигурации датасета
        epochs=50,  # Количество эпох обучения
        patience=5,  # Раннее прекращение после 5 эпох без улучшения
        imgsz=640,  # Размер входных изображений
        batch=5,  # Размер батча
        device="cuda" if torch.cuda.is_available() else "cpu",  # Устройство для вычислений
        augment=True  # Включение аугментации данных
    )

    # Экспорт модели в формат ONNX
    model.export(format="onnx")
    print("✅ Обучение завершено! Модель сохранена.")


# Запуск программы
"""Запуск функции обучения."""
if __name__ == "__main__":
    train_yolo()