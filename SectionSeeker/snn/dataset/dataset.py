import codecs
import os
import os.path
import shutil
import string
import sys
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
import cv2

from torchvision.datasets.utils import (
    _flip_byte_order,
    check_integrity,
    download_and_extract_archive,
    extract_archive,
    verify_str_arg,
)
from torchvision.datasets import VisionDataset
import glob


class SeismicDataset(VisionDataset):
    """Seimic objects dataset for label-efficient learning

    Args:
        root (string): Root directory of dataset
        train (bool, optional): If True, creates dataset ./data/train/ directory, else ./data/test/
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        build (bool, optional): If True, ignore any saved tensors and build dataset from filesystem
        greyscale (bool, optional): If True, load images in grayscale (else use RGB)
        compatibility_mode (bool, optional): If True, return PIL.Image objects as data (instead of torch.Tensor objects if False)
        normalize (bool, optional): If True, images will be normalized image-wise when loaded from disk
    """

    training_file = "training.pt"
    test_file = "test.pt"
    classes = [
        "Boring",
        "Bright_Planar",
        "Bright_Chaotic",
        "Channel",
        "Converging_Amplitudes",
        "Fault",
        "Salt",
        "Transparent_Planar",
    ]

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        build: bool = False,
        greyscale: bool = True,
        compatibility_mode: bool = False,
        normalize: bool = True,
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)
        self.train = train  # training set or test set
        self.greyscale = greyscale
        self.colormode = "L" if greyscale else "RGB"
        self.compatibility_mode = compatibility_mode
        self.normalize = normalize

        if build:
            self.data, self.targets = self._load_from_fs(root)
            return

        self.data, self.targets = self._load_data()

    def _load_data(self):
        image_file = f"{'train' if self.train else 'test'}-images.pt"
        data = self._read_tensor(os.path.join(self.raw_folder, image_file))

        label_file = f"{'train' if self.train else 'test'}-labels.pt"
        targets = self._read_tensor(os.path.join(self.raw_folder, label_file))

        return data, targets

    def _read_tensor(self, filepath):
        return torch.load(filepath)

    def _load_from_fs(self, root: str = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if not root:
            root = os.getcwd()
        dirpath = f"{root}/data/{'train' if self.train else 'test'}"
        data = []
        targets = []
        for c in self.classes:
            images = []
            target = self.class_to_idx.get(c)
            for filename in glob.glob(f"{dirpath}/{c}/*png"):
                im = Image.open(filename)
                # resize to most common size
                im = im.convert(self.colormode).resize((118, 143))
                if self.normalize:
                    im = cv2.normalize(
                        np.array(im), None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32FC1
                    )
                images.append(im)
            if len(images) == 0:
                continue
            data.append(np.array(images, dtype=np.float32))
            targets.append(np.full(len(images), target))

        return torch.from_numpy(np.concatenate(data, axis=0)), torch.from_numpy(
            np.concatenate(targets, axis=0)
        )

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if self.compatibility_mode:
            img = Image.fromarray(img.numpy(), mode=self.colormode)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "processed")

    @property
    def class_to_idx(self) -> Dict[str, int]:
        return {_class: i for i, _class in enumerate(self.classes)}

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"

    def save(self) -> None:
        image_file = f"{'train' if self.train else 'test'}-images.pt"
        torch.save(self.data, os.path.join(self.raw_folder, image_file))

        label_file = f"{'train' if self.train else 'test'}-labels.pt"
        torch.save(self.targets, os.path.join(self.raw_folder, label_file))
