import warnings
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import rasterio as rio
from tqdm import tqdm

warnings.filterwarnings("ignore", category=rio.errors.NotGeoreferencedWarning)


DIR_PATH = Path('../../data/Potsdam/labels')
TARGET_DIR_PATH = Path('../../data/Potsdam_prepared/labels')
# original label size is 6000px x 6000px (.05 m/px), target label size is 1500px x 1500px (.2 m/px)
TARGET_LABEL_SIZE = 1500
CLASS_MAP = {
    (255, 255, 255): 1,  # impervious surfaces
    (0, 0, 255): 2,  # building
    (0, 255, 255): 3,  # low vegetation
    (0, 255, 0): 4,  # tree
    (255, 255, 0): 5,  # car
    (255, 0, 0): 0,  # clutter/ background
}


def resize_label(
    image: np.ndarray,
    target_label_size: int,
) -> np.ndarray:
    return cv2.resize(image, (target_label_size, target_label_size), interpolation=cv2.INTER_NEAREST)


def map_label_values(
    label: npt.NDArray[np.uint8],
    class_map: dict[(int, int, int), int],
) -> npt.NDArray[np.uint8]:
    label_encoded = np.zeros((label.shape[0], label.shape[1]), dtype=np.uint8)  # create array of zeros

    for value_rgb, value_class in class_map.items():  # iterate over class map
        matches = np.all(label == np.array(value_rgb, dtype=np.uint8), axis=-1)  # find matches
        label_encoded[matches] = value_class  # assign class value to matches

    return label_encoded


def prepare_label(
    file_path: Path,
    target_dir_path: Path,
    target_label_size: int = None,
) -> None:
    with rio.open(file_path) as src:
        label = src.read()
        label = label.transpose(1, 2, 0)  # channels first to channels last

        if target_label_size is not None:
            label = resize_label(label, target_label_size)

        label = map_label_values(
            label=label,
            class_map=CLASS_MAP,
        )

        target_file_path = target_dir_path / file_path.with_suffix('.npy').name
        np.save(target_file_path, label)


def prepare_labels(
    dir_path: Path,
    target_dir_path: Path,
    target_label_size: int = None,
) -> None:
    file_paths = [file_path for file_path in list(dir_path.iterdir()) if file_path.suffix == '.tif']

    for file_path in tqdm(file_paths):
        prepare_label(
            file_path=file_path,
            target_dir_path=target_dir_path,
            target_label_size=target_label_size,
        )


if __name__ == '__main__':
    prepare_labels(
        dir_path=DIR_PATH,
        target_dir_path=TARGET_DIR_PATH,
        target_label_size=TARGET_LABEL_SIZE,
    )
