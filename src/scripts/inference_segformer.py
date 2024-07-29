from pathlib import Path

import lightning as L
import torch

from src.datamodule import DataModule
from src.dataset import Dataset
from src.model_segformer import MulticlassSegFormer
from src.model_config import SegFormerConfigFactory

CKPT_PATH = Path('../../data/checkpoints/segformer_b0.ckpt')

mean_values = [.485 * 255, .456 * 255, .406 * 255, .5 * 255]  # ImageNet
std_values = [.229 * 255, .224 * 255, .225 * 255, .5 * 255]  # ImageNet

predict_dataset = Dataset.from_pickles(
    pickle_images_path=Path('../../data/Potsdam_prepared/images_val_paths.pkl'),
    sample_size=(1500, 1500),
    range_values=255,
    mean_values=mean_values,
    std_values=std_values,
    standardization=True,
    num_classes=6,
    one_hot_encoding=True,
)

datamodule = DataModule(
    predict_dataset=predict_dataset,
    batch_size=8,
    num_workers=8,
    pin_memory=True,
    prefetch_factor=2,
)

config_factory = SegFormerConfigFactory(
    num_channels=4,
    num_classes=6,
)

labels = ['background', 'impervious_surfaces', 'building', 'low_vegetation', 'tree', 'car']

model = MulticlassSegFormer.load_from_checkpoint(
    CKPT_PATH,
    config=config_factory.b0_config,
    weights=CKPT_PATH,
    num_classes=6,
    labels=labels,
    criterion=torch.nn.CrossEntropyLoss(),
    learning_rate=1e-4,
    weight_decay=1e-2,
    prediction_dir_path=Path('../../data/predictions/segformer'),
)

trainer = L.Trainer(
    accelerator='cpu',
    precision='16-mixed',
)

trainer.predict(
    model=model,
    datamodule=datamodule,
)
