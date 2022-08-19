from typing import Optional, Callable, Tuple, Any

import copy
import random
from numpy import dtype
import torch
import torch.nn
import torch.utils.data

from torchvision.datasets import FashionMNIST


def partial_remove(data, targets, ratio):
    tmp = list(zip(data, targets))
    partial_tmp = random.sample(tmp, int(len(tmp) * ratio))
    new_data = [ d[0] for d in partial_tmp ]
    new_targets = [ d[1] for d in partial_tmp ]
    return new_data, new_targets


class PartialFashionMNIST(FashionMNIST):

    def __init__(self,
            root: str,
            partial: float = 1.0,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False
    ) -> None:
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        if not train:
            return

        self.data, self.targets = partial_remove(self.data, self.targets, partial)


class MaskFashionMNIST(FashionMNIST):

    def __init__(self,
            root: str,
            partial: float = -1.0,
            ratio: float = 1.0,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False
    ) -> None:
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        if not train:
            return

        if 0 < partial < 1:
            self.data, self.targets = partial_remove(self.data, self.targets, partial)

        tmp = list(zip(self.data, self.targets))
        num_all = len(tmp)
        num_label = int(num_all * ratio)
        num_unlabel = num_all - num_label
        self.mask = [1] * num_label + [0] * num_unlabel
        random.shuffle(self.mask)

        max_target = max(self.targets)
        for idx, m in enumerate(self.mask):
            if m == 0:
                self.targets[idx] = random.randint(0, max_target)


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target =  super().__getitem__(index)
        m = self.mask[index]
        return img, target, m


def mask_dataset_to_label_dataset(dataset: MaskFashionMNIST):
    new_dataset = copy.deepcopy(dataset)
    new_dataset.data = [ dataset.data[i] for i, m in enumerate(dataset.mask) if m > 0.5 ]
    new_dataset.targets = [ dataset.targets[i] for i, m in enumerate(dataset.mask) if m > 0.5 ]
    return new_dataset


def split_dataset(dataset: FashionMNIST, ratio):

    data, targets = dataset.data, dataset.targets
    total_len = len(dataset)
    idxs = set(range(total_len))
    label_idxs = set(random.sample(idxs, int(total_len * ratio)))
    unlabel_idxs = idxs - label_idxs

    label_dataset = copy.deepcopy(dataset)
    unlabel_dataset = copy.deepcopy(dataset)

    label_dataset.data = [ data[i] for i in label_idxs ]
    label_dataset.targets = [ targets[i] for i in label_idxs ]

    unlabel_dataset.data = [ data[i] for i in unlabel_idxs ]
    unlabel_dataset.targets = [ targets[i] for i in unlabel_idxs ]

    return label_dataset, unlabel_dataset


def update_label_dataset(label_dataset: FashionMNIST, unlabel_dataset: FashionMNIST, predict_labels: torch.Tensor, thresh: float = 0.99):
    predict_labels = torch.nn.Softmax(dim=1)(predict_labels)
    max_vals, max_idxs = predict_labels.max(dim=1)
    new_samples_idxs = max_vals > thresh

    new_samples = [ unlabel_dataset.data[i] for i, flag in enumerate(new_samples_idxs) if flag ]
    new_samples_label = [ max_idxs[i] for i, flag in enumerate(new_samples_idxs) if flag ]

    new_dataset = copy.deepcopy(label_dataset)
    new_dataset.data += new_samples
    new_dataset.targets += new_samples_label

    remain_samples = [ unlabel_dataset.data[i] for i, flag in enumerate(new_samples_idxs) if not flag ]
    remain_samples_label = [ max_idxs[i] for i, flag in enumerate(new_samples_idxs) if not flag ]

    remain_dataset = copy.deepcopy(unlabel_dataset)
    remain_dataset.data = remain_samples
    remain_dataset.targets = remain_samples_label

    return new_dataset, remain_dataset


def update_mask_dataset(dataset: MaskFashionMNIST, predict_labels: torch.Tensor):
    predict_labels = torch.nn.Softmax(dim=1)(predict_labels)
    max_vals, new_labels = predict_labels.max(dim=1)

    new_dataset = copy.deepcopy(dataset)
    for idx, m in enumerate(dataset.mask):
        if m < 0.5:
            new_dataset.targets[idx] = new_labels[idx]
    return new_dataset


def noise_dataset(dataset: FashionMNIST, sample_ratio: float, noise_ratio: float):

    def noise_sample(sample, ratio):
        h, w = sample.size()
        num_noise = int(h * w * ratio)
        num_origin = h * w - num_noise
        pixel_idxs = [False] * num_origin + [True] * num_noise
        random.shuffle(pixel_idxs)
        pixel_idxs = torch.Tensor(pixel_idxs).bool().view(h, w)
        sample[pixel_idxs] = 0
        return sample

    num_total_sample = len(dataset)
    num_noise_sample = int(num_total_sample * sample_ratio)
    samples_idxs = list(range(num_total_sample))
    noise_samples_idxs = random.sample(samples_idxs, k=num_noise_sample)

    new_dataset = copy.deepcopy(dataset)
    for idx in noise_samples_idxs:
        new_dataset.data[idx] = noise_sample(dataset.data[idx], noise_ratio)

    return new_dataset


def mislabel_dataset(dataset: FashionMNIST, sample_ratio: float):

    num_total_sample = len(dataset)
    num_noise_sample = int(num_total_sample * sample_ratio)
    samples_idxs = list(range(num_total_sample))
    noise_samples_idxs = random.sample(samples_idxs, k=num_noise_sample)

    max_label = max(dataset.targets)

    new_dataset = copy.deepcopy(dataset)
    for idx in noise_samples_idxs:
        new_dataset.targets[idx] = random.randint(0, max_label)

    return new_dataset
