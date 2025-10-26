import json
import os
from typing import Dict, Any
import torch
from detectron2.config import CfgNode
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.model_zoo import model_zoo
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

# Регистрация датасета
"""Регистрация обучающего и валидационного датасетов в формате COCO."""
data_dir: str = "RetinaNetDataset"  # Директория с данными
for split in ["train", "val"]:
    register_coco_instances(
        f"my_dataset_{split}",
        {"thing_classes": ["bottle", "cream", "kefir", "ketchup", "mayonnaise", "milk", "milkcream", "mustard",
                           "ryazhenka", "sourcream"]},
        f"{data_dir}/{split}/result_{split}.json",
        f"{data_dir}/{split}/images"
    )

# Проверка регистрации датасета
dataset_dicts: Dict[str, Any] = DatasetCatalog.get("my_dataset_train")  # Загрузка данных обучающего датасета
metadata: Any = MetadataCatalog.get("my_dataset_train")  # Получение метаданных датасета
print("Метаданные:", metadata)

# Конфигурация модели
"""Настройка конфигурации модели RetinaNet."""
cfg: CfgNode = get_cfg()  # Создание объекта конфигурации
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_1x.yaml"))  # Загрузка конфигурации RetinaNet
cfg.DATASETS.TRAIN = ("my_dataset_train",)  # Указание обучающего датасета
cfg.DATASETS.TEST = ("my_dataset_val",)  # Указание валидационного датасета
cfg.DATALOADER.NUM_WORKERS = 4  # Количество рабочих процессов для загрузки данных
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_50_FPN_1x.yaml")  # Предобученные веса
cfg.SOLVER.IMS_PER_BATCH = 5  # Размер батча
cfg.SOLVER.BASE_LR = 0.01  # Базовая скорость обучения
cfg.SOLVER.MAX_ITER = 1000  # Максимальное количество итераций
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # Размер батча для ROI heads
cfg.MODEL.RETINANET.NUM_CLASSES = len(metadata.thing_classes)  # Количество классов для детекции


# Основная функция обучения и оценки
def main() -> None:
    """
    Основная функция для обучения и оценки модели RetinaNet.
    """
    print(f"Количество классов: {len(metadata.thing_classes)}")

    # Установка устройства для вычислений
    if torch.cuda.is_available():
        print("Используется CUDA")
        cfg.MODEL.DEVICE = "cuda"
    else:
        print("Используется CPU")
        cfg.MODEL.DEVICE = "cpu"

    # Инициализация и обучение модели
    trainer: DefaultTrainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)  # Загрузка модели без возобновления обучения
    trainer.train()  # Запуск обучения

    # Оценка модели на валидационной выборке
    cfg.OUTPUT_DIR = "bestRetinaNetModel/"  # Директория для сохранения результатов
    evaluator: COCOEvaluator = COCOEvaluator("my_dataset_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "my_dataset_val")  # Загрузчик валидационных данных
    inference_on_dataset(trainer.model, val_loader, evaluator)  # Выполнение оценки

    # Сохранение результатов
    output_dir: str = cfg.OUTPUT_DIR
    print(f"Модель сохранена в {output_dir}")

    # Сохранение конфигурации
    config_output_path: str = os.path.join(output_dir, "config.yaml")
    with open(config_output_path, "w") as f:
        f.write(cfg.dump())
    print(f"Конфигурация сохранена в {config_output_path}")

    # Сохранение метаданных
    metadata_path: str = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump({"thing_classes": metadata.thing_classes}, f)
    print(f"Метаданные сохранены в {metadata_path}")


# Запуск программы
"""Запуск основной функции."""
if __name__ == '__main__':
    main()