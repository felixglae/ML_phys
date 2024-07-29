from pathlib import Path

import cv2
import numpy as np
import rasterio as rio
from tqdm import tqdm


DIR_PATH = Path('../../data/Potsdam/rgbi_images')
TARGET_DIR_PATH = Path('../../data/Potsdam_prepared/rgbi_images')
# original image size is 6000px x 6000px (.05 m/px), target image size is 1500px x 1500px (.2 m/px)
TARGET_IMAGE_SIZE = 1500


def resize_image(
    image: np.ndarray,
    target_image_size: int,
) -> np.ndarray:
    return cv2.resize(image, (target_image_size, target_image_size), interpolation=cv2.INTER_LINEAR)


def prepare_image(
    file_path: Path,
    target_dir_path: Path,
    target_image_size: int = None,
) -> None:
    with rio.open(file_path) as src:
        image = src.read()
        image = image.transpose(1, 2, 0)  # channels first to channels last

        if target_image_size is not None:
            image = resize_image(image, target_image_size)

        target_file_path = target_dir_path / file_path.with_suffix('.npy').name
        np.save(target_file_path, image)


def prepare_images(
    dir_path: Path,
    target_dir_path: Path,
    target_image_size: int = None,
) -> None:
    file_paths = [file_path for file_path in list(dir_path.iterdir()) if file_path.suffix == '.tif']

    for file_path in tqdm(file_paths):
        prepare_image(
            file_path=file_path,
            target_dir_path=target_dir_path,
            target_image_size=target_image_size,
        )


if __name__ == '__main__':
    prepare_images(
        dir_path=DIR_PATH,
        target_dir_path=TARGET_DIR_PATH,
        target_image_size=TARGET_IMAGE_SIZE,
    )
