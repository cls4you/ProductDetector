import os
from typing import Tuple, Dict, List, Any
import torch
import torchvision
from transformers import DetrForObjectDetection, DetrImageProcessor
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# Настройка путей и параметров
"""Настройка путей к данным и параметров модели."""
HOME: str = os.getcwd()  # Текущая рабочая директория
DATASET_PATH: str = os.path.join(HOME, "DetrDataset")  # Путь к датасету
TRAIN_DIRECTORY: str = os.path.join(DATASET_PATH, "train")  # Директория обучающих данных
VAL_DIRECTORY: str = os.path.join(DATASET_PATH, "val")  # Директория валидационных данных
TRAIN_ANNOTATION_FILE: str = "result_train.json"  # Файл аннотаций для обучения
VAL_ANNOTATION_FILE: str = "result_val.json"  # Файл аннотаций для валидации
TRAIN_IMAGES_DIR: str = os.path.join(TRAIN_DIRECTORY, "images")  # Директория изображений для обучения
VAL_IMAGES_DIR: str = os.path.join(VAL_DIRECTORY, "images")  # Директория изображений для валидации

DEVICE: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # Устройство для вычислений
CHECKPOINT: str = 'facebook/detr-resnet-50'  # Предобученная модель Detr
CONFIDENCE_THRESHOLD: float = 0.5  # Порог уверенности для предсказаний

# Инициализация процессора изображений
image_processor: DetrImageProcessor = DetrImageProcessor.from_pretrained(CHECKPOINT)


# Определение пользовательского датасета
class CocoDetection(torchvision.datasets.CocoDetection):
    """Класс для работы с датасетом в формате COCO для Detr."""

    def __init__(self, image_directory_path: str, image_processor: DetrImageProcessor,
                 annotation_file: str, train: bool = True) -> None:
        """
        Инициализация датасета COCO.

        Args:
            image_directory_path (str): Путь к директории с данными.
            image_processor (DetrImageProcessor): Процессор изображений для Detr.
            annotation_file (str): Имя файла аннотаций.
            train (bool): Флаг, указывающий, является ли датасет обучающим.
        """
        annotation_file_path: str = os.path.join(image_directory_path, annotation_file)
        images_path: str = os.path.join(image_directory_path, "images")
        super().__init__(images_path, annotation_file_path)
        self.image_processor: DetrImageProcessor = image_processor
        self.root: str = images_path

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Получение элемента датасета по индексу.

        Args:
            idx (int): Индекс элемента.

        Returns:
            Tuple[torch.Tensor, Dict[str, Any]]: Пиксельные значения изображения и аннотации.
        """
        images, annotations = super().__getitem__(idx)
        image_id: int = self.ids[idx]
        annotations: Dict[str, Any] = {'image_id': image_id, 'annotations': annotations}
        encoding: Dict[str, Any] = self.image_processor(images=images, annotations=annotations, return_tensors="pt")
        pixel_values: torch.Tensor = encoding["pixel_values"].squeeze()
        target: Dict[str, Any] = encoding["labels"][0]
        return pixel_values, target


# Создание датасетов и загрузчиков данных
"""Инициализация обучающего и валидационного датасетов."""
TRAIN_DATASET: CocoDetection = CocoDetection(
    image_directory_path=TRAIN_DIRECTORY,
    image_processor=image_processor,
    annotation_file=TRAIN_ANNOTATION_FILE,
    train=True
)
VAL_DATASET: CocoDetection = CocoDetection(
    image_directory_path=VAL_DIRECTORY,
    image_processor=image_processor,
    annotation_file=VAL_ANNOTATION_FILE,
    train=False
)

print(f"Количество обучающих примеров: {len(TRAIN_DATASET)}")
print(f"Количество валидационных примеров: {len(VAL_DATASET)}")


# Функция для обработки батчей
def collate_fn(batch: List[Tuple[torch.Tensor, Dict]]) -> Dict[str, Any]:
    """
    Обработка батча для DataLoader.

    Args:
        batch (List[Tuple[torch.Tensor, Dict]]): Список элементов батча.

    Returns:
        Dict[str, Any]: Обработанные данные батча (пиксельные значения, маски и метки).
    """
    pixel_values: List[torch.Tensor] = [item[0] for item in batch]
    encoding: Dict[str, Any] = image_processor.pad(pixel_values, return_tensors="pt")
    labels: List[Dict] = [item[1] for item in batch]
    return {
        'pixel_values': encoding['pixel_values'],
        'pixel_mask': encoding['pixel_mask'],
        'labels': labels
    }


# Создание загрузчиков данных
TRAIN_DATALOADER: DataLoader = DataLoader(
    dataset=TRAIN_DATASET,
    collate_fn=collate_fn,
    batch_size=5,
    shuffle=True
)
VAL_DATALOADER: DataLoader = DataLoader(
    dataset=VAL_DATASET,
    collate_fn=collate_fn,
    batch_size=5
)


# Определение модели Detr
class Detr(pl.LightningModule):
    """Модель Detr на основе PyTorch Lightning для обучения и валидации."""

    def __init__(self, lr: float, lr_backbone: float, weight_decay: float) -> None:
        """
        Инициализация модели Detr.

        Args:
            lr (float): Скорость обучения для модели.
            lr_backbone (float): Скорость обучения для backbone.
            weight_decay (float): Коэффициент регуляризации.
        """
        super().__init__()
        self.model: DetrForObjectDetection = DetrForObjectDetection.from_pretrained(
            pretrained_model_name_or_path=CHECKPOINT,
            num_labels=len(TRAIN_DATASET.coco.cats),
            ignore_mismatched_sizes=True
        )

        # Обновление меток категорий на основе датасета
        custom_categories: Dict = TRAIN_DATASET.coco.cats
        new_id2label: Dict[int, str] = {
            cat_id: custom_categories[cat_id]['name']
            for cat_id in sorted(custom_categories.keys())
        }
        self.model.config.id2label = new_id2label
        self.model.config.label2id = {v: k for k, v in self.model.config.id2label.items()}

        self.lr: float = lr
        self.lr_backbone: float = lr_backbone
        self.weight_decay: float = weight_decay

    def forward(self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor) -> Any:
        """
        Прямой проход модели.

        Args:
            pixel_values (torch.Tensor): Пиксельные значения изображений.
            pixel_mask (torch.Tensor): Маска для пиксельных значений.

        Returns:
            Any: Результаты прямого прохода модели.
        """
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    def common_step(self, batch: Dict[str, Any], batch_idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Общий шаг для обучения и валидации.

        Args:
            batch (Dict[str, Any]): Батч данных.
            batch_idx (int): Индекс батча.

        Returns:
            Tuple[torch.Tensor, Dict[str, Any]]: Значение потерь и словарь с потерями.
        """
        pixel_values: torch.Tensor = batch["pixel_values"]
        pixel_mask: torch.Tensor = batch["pixel_mask"]
        labels: List[Dict] = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        loss: torch.Tensor = outputs.loss
        loss_dict: Dict[str, Any] = outputs.loss_dict
        return loss, loss_dict

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """
        Шаг обучения модели.

        Args:
            batch (Dict[str, Any]): Батч данных.
            batch_idx (int): Индекс батча.

        Returns:
            torch.Tensor: Значение потерь.
        """
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("training_loss", loss)
        for k, v in loss_dict.items():
            self.log(f"train_{k}", v.item())
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """
        Шаг валидации модели.

        Args:
            batch (Dict[str, Any]): Батч данных.
            batch_idx (int): Индекс батча.

        Returns:
            torch.Tensor: Значение потерь.
        """
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation/loss", loss)
        for k, v in loss_dict.items():
            self.log(f"validation_{k}", v.item())
        return loss

    def configure_optimizers(self) -> torch.optim.AdamW:
        """
        Настройка оптимизатора.

        Returns:
            torch.optim.AdamW: Оптимизатор AdamW.
        """
        param_dicts = [
            {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {"params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
             "lr": self.lr_backbone},
        ]
        return torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)

    def train_dataloader(self) -> DataLoader:
        """
        Получение загрузчика данных для обучения.

        Returns:
            DataLoader: Загрузчик обучающих данных.
        """
        return TRAIN_DATALOADER

    def val_dataloader(self) -> DataLoader:
        """
        Получение загрузчика данных для валидации.

        Returns:
            DataLoader: Загрузчик валидационных данных.
        """
        return VAL_DATALOADER


# Обучение и сохранение модели
"""Инициализация модели, обучение и сохранение результатов."""
model: Detr = Detr(lr=0.01, lr_backbone=0.01, weight_decay=1e-4)
trainer: pl.Trainer = pl.Trainer(
    devices=1,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    max_epochs=50,
    gradient_clip_val=0.1,
    accumulate_grad_batches=8,
    log_every_n_steps=5
)

# Запуск обучения
trainer.fit(model)

# Сохранение модели и процессора
MODEL_PATH: str = os.path.join(HOME, 'detr-model')
model.model.save_pretrained(MODEL_PATH)
image_processor.save_pretrained(MODEL_PATH)

# Вывод меток для проверки
print("Сохранённые метки id2label:", model.model.config.id2label)