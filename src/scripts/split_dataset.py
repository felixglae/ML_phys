import pickle
import random
from pathlib import Path


IMAGES_DIR_PATH = Path('../../data/Potsdam_prepared/rgbi_images')
LABELS_DIR_PATH = Path('../../data/Potsdam_prepared/labels')


def split_dataset(
    images_dir_path: Path,
    labels_dir_path: Path,
    train_ratio: float = .8,
) -> None:
    sample_indices_range = list(range(0, len(list(images_dir_path.iterdir()))))
    num_train_samples = int(len(sample_indices_range) * train_ratio)

    random.shuffle(sample_indices_range)

    train_indices = sample_indices_range[:num_train_samples]
    val_indices = sample_indices_range[num_train_samples:]

    images_train_paths = [sorted(list(images_dir_path.iterdir()))[i] for i in train_indices]
    with open('../../data/Potsdam_prepared/images_train_paths.pkl', 'wb') as f:
        pickle.dump(images_train_paths, f)

    images_val_paths = [sorted(list(images_dir_path.iterdir()))[i] for i in val_indices]
    with open('../../data/Potsdam_prepared/images_val_paths.pkl', 'wb') as f:
        pickle.dump(images_val_paths, f)

    labels_train_paths = [sorted(list(labels_dir_path.iterdir()))[i] for i in train_indices]
    with open('../../data/Potsdam_prepared/labels_train_paths.pkl', 'wb') as f:
        pickle.dump(labels_train_paths, f)

    labels_val_paths = [sorted(list(labels_dir_path.iterdir()))[i] for i in val_indices]
    with open('../../data/Potsdam_prepared/labels_val_paths.pkl', 'wb') as f:
        pickle.dump(labels_val_paths, f)


if __name__ == '__main__':
    split_dataset(
        images_dir_path=IMAGES_DIR_PATH,
        labels_dir_path=LABELS_DIR_PATH,
    )
