import pickle
from pathlib import Path

import albumentations as A
import numpy as np
import numpy.typing as npt
import torch.utils.data


class Dataset(torch.utils.data.Dataset):

    def __init__(
        self,
        images_paths: list[Path],
        labels_paths: list[Path] = None,
        sample_size: tuple[int, int] = None,
        patch_size: tuple[int, int] = None,
        stride: tuple[int, int] = None,
        padding: bool = False,
        augmentation_per_channel: list[tuple[A.Compose | None, int]] = None,
        augmentation: A.Compose = None,
        range_values: int | float = None,
        normalization: bool = False,
        mean_values: list[float] = None,
        std_values: list[float] = None,
        standardization: bool = False,
        num_classes: int = None,
        one_hot_encoding: bool = False,
    ) -> None:
        self.images_paths = self._cast_paths(images_paths)
        self.labels_paths = self._cast_paths(labels_paths) if labels_paths is not None else None
        self.sample_size = sample_size
        self.patch_size = patch_size
        self.stride = stride if stride is not None else patch_size
        self.padding = padding
        self.augmentation_per_channel = augmentation_per_channel
        self.augmentation = augmentation
        self.range_values = float(range_values) if range_values is not None else None
        self.normalization = normalization if standardization is False else False
        self.mean_values = np.array(mean_values) if mean_values is not None else None
        self.std_values = np.array(std_values) if std_values is not None else None
        self.standardization = standardization
        self.num_classes = num_classes
        self.one_hot_encoding = one_hot_encoding

        self._num_samples = self._calculate_num_samples()
        self._pad_size = self._calculate_pad_size()
        self._num_patches_per_sample = self._calculate_num_patches_per_sample()

    @staticmethod
    def _cast_paths(paths: list[Path] | None) -> npt.NDArray | None:
        if paths is None:
            return None

        paths = [str(path) for path in paths]
        return np.array(paths).astype(np.string_)

    def _calculate_num_samples(self) -> int:
        return len(self.images_paths)

    def _calculate_pad_size(self) -> tuple[int, int]:
        if self.padding is False:
            return 0, 0

        sample_height, sample_width = self.sample_size
        patch_height, patch_width = self.patch_size
        stride_height, stride_width = self.stride

        sample_height_remainder = (sample_height - patch_height) % stride_height
        sample_width_remainder = (sample_width - patch_width) % stride_width

        pad_height = (stride_height - sample_height_remainder) % stride_height
        pad_width = (stride_width - sample_width_remainder) % stride_width

        return pad_height, pad_width

    def _calculate_num_patches_per_sample(self) -> int:
        if self.patch_size is None:
            return 1

        sample_height, sample_width = self.sample_size
        patch_height, patch_width = self.patch_size
        stride_height, stride_width = self.stride
        pad_height, pad_width = self._pad_size

        if self.padding:
            sample_height += pad_height
            sample_width += pad_width

        num_patches_height = (sample_height - patch_height) // stride_height + 1
        num_patches_width = (sample_width - patch_width) // stride_width + 1

        return num_patches_height * num_patches_width

    @classmethod
    def from_dirs(
        cls,
        images_dir_path: Path,
        labels_dir_path: Path = None,
        sample_size: tuple[int, int] = None,
        patch_size: tuple[int, int] = None,
        stride: tuple[int, int] = None,
        padding: bool = False,
        augmentation_per_channel: list[tuple[A.Compose | None, int]] = None,
        augmentation: A.Compose = None,
        range_values: int | float = None,
        normalization: bool = False,
        mean_values: list[float] = None,
        std_values: list[float] = None,
        standardization: bool = False,
        num_classes: int = None,
        one_hot_encoding: bool = False,
    ) -> 'Dataset':
        images_paths = cls._gather_paths(images_dir_path)
        labels_paths = cls._gather_paths(labels_dir_path) if labels_dir_path is not None else None

        return cls(
            images_paths=images_paths,
            labels_paths=labels_paths,
            sample_size=sample_size,
            patch_size=patch_size,
            stride=stride,
            padding=padding,
            augmentation_per_channel=augmentation_per_channel,
            augmentation=augmentation,
            range_values=range_values,
            normalization=normalization,
            mean_values=mean_values,
            std_values=std_values,
            standardization=standardization,
            num_classes=num_classes,
            one_hot_encoding=one_hot_encoding,
        )

    @staticmethod
    def _gather_paths(dir_path: Path) -> list[Path]:
        return sorted(list(dir_path.iterdir()))

    @classmethod
    def from_pickles(
        cls,
        pickle_images_path: Path,
        pickle_labels_path: Path = None,
        sample_size: tuple[int, int] = None,
        patch_size: tuple[int, int] = None,
        stride: tuple[int, int] = None,
        padding: bool = False,
        augmentation_per_channel: list[tuple[A.Compose | None, int]] = None,
        augmentation: A.Compose = None,
        range_values: int | float = None,
        normalization: bool = False,
        mean_values: list[float] = None,
        std_values: list[float] = None,
        standardization: bool = False,
        num_classes: int = None,
        one_hot_encoding: bool = False,
    ) -> 'Dataset':
        images_paths = cls._load_pickle(pickle_images_path)
        labels_paths = cls._load_pickle(pickle_labels_path) if pickle_labels_path is not None else None

        return cls(
            images_paths=images_paths,
            labels_paths=labels_paths,
            sample_size=sample_size,
            patch_size=patch_size,
            stride=stride,
            padding=padding,
            augmentation_per_channel=augmentation_per_channel,
            augmentation=augmentation,
            range_values=range_values,
            normalization=normalization,
            mean_values=mean_values,
            std_values=std_values,
            standardization=standardization,
            num_classes=num_classes,
            one_hot_encoding=one_hot_encoding,
        )

    @staticmethod
    def _load_pickle(pickle_path: Path) -> list[Path]:
        with open(pickle_path, 'rb') as file:
            return pickle.load(file)

    def __len__(self) -> int:
        return self._num_samples * self._num_patches_per_sample

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        sample_index = index // self._num_patches_per_sample
        patch_index = index % self._num_patches_per_sample

        image, label = self._load_npy(sample_index)
        image, label = self._preprocess(image, label, patch_index)
        image, label = self._to_tensor(image, label)

        if label is not None:
            return image, label

        return image

    def _load_npy(
        self,
        sample_index: int,
    ) -> tuple[npt.NDArray, npt.NDArray] | tuple[npt.NDArray, None]:
        image_path = Path(str(self.images_paths[sample_index], encoding='utf-8'))
        image = np.load(image_path)

        if self.labels_paths is not None:
            label_path = Path(str(self.labels_paths[sample_index], encoding='utf-8'))
            label = np.load(label_path)
        else:
            label = None

        return image, label

    def _preprocess(
        self,
        image: npt.NDArray,
        label: npt.NDArray | None,
        patch_index: int,
    ) -> tuple[npt.NDArray, npt.NDArray] | tuple[npt.NDArray, None]:
        if self._num_patches_per_sample > 1:
            image, label = self._patch(image, label, patch_index)

        if self.augmentation_per_channel is not None:
            image = self._augment_per_channel(image)

        if self.augmentation is not None:
            image, label = self._augment(image, label)

        if self.normalization:
            image = self._normalize(image)

        if self.standardization:
            image = self._standardize(image)

        if self.one_hot_encoding and label is not None:
            label = self._one_hot(label)

        return image, label

    def _patch(
        self,
        image: npt.NDArray,
        label: npt.NDArray | None,
        patch_index: int,
    ) -> tuple[npt.NDArray, npt.NDArray] | tuple[npt.NDArray, None]:
        _, sample_width = self.sample_size
        _, pad_width = self._pad_size
        sample_width += pad_width
        patch_height, patch_width = self.patch_size
        stride_height, stride_width = self.stride

        if self.padding:
            image = self._pad(image)
            label = self._pad(label) if label is not None else None

        num_patches_width = (sample_width - patch_width) // stride_width + 1
        row_index = patch_index // num_patches_width
        col_index = patch_index % num_patches_width

        y = row_index * stride_height
        x = col_index * stride_width

        image = image[y:y + patch_height, x:x + patch_width]
        label = label[y:y + patch_height, x:x + patch_width] if label is not None else None

        return image, label

    def _pad(
        self,
        array: npt.NDArray,
    ) -> npt.NDArray:
        pad_height, pad_width = self._pad_size

        if pad_height == 0 and pad_width == 0:
            return array

        if array.ndim == 2:
            return np.pad(array,
                          pad_width=((0, pad_height), (0, pad_width)),
                          mode='constant')

        return np.pad(array,
                      pad_width=((0, pad_height), (0, pad_width), (0, 0)),
                      mode='constant')

    def _augment_per_channel(
        self,
        image: npt.NDArray,
    ) -> npt.NDArray:
        channel_index = 0

        for augmentation, num_channels in self.augmentation_per_channel:
            channels = image[..., channel_index:channel_index + num_channels]

            if augmentation is not None:
                augmented = augmentation(image=channels)
                image[..., channel_index:channel_index + num_channels] = augmented['image']

            channel_index += num_channels

        return image

    def _augment(
        self,
        image: npt.NDArray,
        label: npt.NDArray | None,
    ) -> tuple[npt.NDArray, npt.NDArray] | tuple[npt.NDArray, None]:
        if label is not None:
            augmented = self.augmentation(image=image, mask=label)
        else:
            augmented = self.augmentation(image=image)

        image = augmented['image']
        label = augmented['mask'] if label is not None else None

        return image, label

    def _normalize(
        self,
        image: npt.NDArray,
    ) -> npt.NDArray:
        return image / self.range_values

    def _standardize(
        self,
        image: npt.NDArray,
    ) -> npt.NDArray:
        return ((image / self.range_values - self.mean_values / self.range_values)
            / (self.std_values / self.range_values))

    def _one_hot(
        self,
        label: npt.NDArray,
    ) -> npt.NDArray:
        return np.eye(self.num_classes)[label]

    def _to_tensor(
        self,
        image: npt.NDArray,
        label: npt.NDArray | None,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, None]:
        image = torch.from_numpy(image).float().permute(2, 0, 1)  # channels last to channels first

        if label is not None:
            label = torch.from_numpy(label).float()
            if self.one_hot_encoding:
                label = label.permute(2, 0, 1)  # channels last to channels first

        return image, label
