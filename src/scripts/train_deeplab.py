import os
from pathlib import Path

import albumentations as A
import cv2
import lightning as L
import lightning.pytorch.callbacks
import lightning.pytorch.loggers
import torch
from dotenv import load_dotenv
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights

from src.datamodule import DataModule
from src.dataset import Dataset
from src.model_deeplab import MulticlassDeepLab

load_dotenv()
NEPTUNE_API_TOKEN = os.getenv('NEPTUNE_API_TOKEN')

torch.set_float32_matmul_precision('high')

augmentation_rgb = A.Compose([
    A.HueSaturationValue(p=.5),
])

augmentation_nir = A.Compose([
])

augmentation_per_channel = [(augmentation_rgb, 3), (augmentation_nir, 1)]

augmentation = A.Compose([
    A.GaussNoise(p=.5),
    A.RandomBrightnessContrast(brightness_limit=.1, contrast_limit=.1, p=.5),
    A.Sharpen(alpha=(.2, .5), lightness=(.8, 1.), p=.5),
    A.Rotate(limit=180, border_mode=cv2.BORDER_CONSTANT, p=1.),
    A.HorizontalFlip(p=.5),
])

mean_values = [.485 * 255, .456 * 255, .406 * 255, .5 * 255]  # ImageNet
std_values = [.229 * 255, .224 * 255, .225 * 255, .5 * 255]  # ImageNet

train_dataset = Dataset.from_pickles(
    pickle_images_path=Path('../../data/Potsdam_prepared/images_train_paths.pkl'),
    pickle_labels_path=Path('../../data/Potsdam_prepared/labels_train_paths.pkl'),
    sample_size=(1500, 1500),
    patch_size=(768, 768),
    stride=(384, 384),
    padding=True,
    augmentation_per_channel=augmentation_per_channel,
    augmentation=augmentation,
    range_values=255,
    mean_values=mean_values,
    std_values=std_values,
    standardization=True,
    num_classes=6,
    one_hot_encoding=True,
)

val_dataset = Dataset.from_pickles(
    pickle_images_path=Path('../../data/Potsdam_prepared/images_val_paths.pkl'),
    pickle_labels_path=Path('../../data/Potsdam_prepared/labels_val_paths.pkl'),
    sample_size=(1500, 1500),
    patch_size=(768, 768),
    stride=(384, 384),
    padding=True,
    range_values=255,
    mean_values=mean_values,
    std_values=std_values,
    standardization=True,
    num_classes=6,
    one_hot_encoding=True,
)

datamodule = DataModule(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    batch_size=8,
    num_workers=8,
    pin_memory=True,
    prefetch_factor=2,
)

labels = ['background', 'impervious_surfaces', 'building', 'low_vegetation', 'tree', 'car']

model = MulticlassDeepLab(
    weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1,  # COCO with VOC labels
    num_channels=4,
    num_classes=6,
    labels=labels,
    criterion=torch.nn.CrossEntropyLoss(),
    learning_rate=1e-4,
    weight_decay=1e-2,
)

logger = L.pytorch.loggers.NeptuneLogger(
    api_key=NEPTUNE_API_TOKEN,
    project='felix.glaesemann/ML-Physik',
    tags=['rgbi', 'deeplab', 'all_classes'],
    source_files=[],
    capture_stdout=False,
    git_ref=False,
)


trainer = L.Trainer(
    accelerator='gpu',
    devices=[0],
    precision='16-mixed',
    logger=logger,
    callbacks=[
        L.pytorch.callbacks.EarlyStopping(monitor='val/jaccard_index_all', patience=20, mode='max'),
        L.pytorch.callbacks.ModelCheckpoint(monitor='val/jaccard_index_all', mode='max'),
    ],
    max_epochs=1000,
    log_every_n_steps=5,
    detect_anomaly=True,
)

trainer.fit(
    model=model,
    datamodule=datamodule,
)
