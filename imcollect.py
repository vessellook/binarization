import warnings
from pathlib import Path

import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img


def _key(f):
    return f.name.split('.')[0].split('_')[0]


def _collect_original_gt_pairs(root):
    pairs = []
    for gt_dir in Path(root).glob('**/gt'):
        if not gt_dir.is_dir():
            continue
        original_dir = gt_dir.parent / 'original'
        if not original_dir.exists() or not original_dir.is_dir():
            continue
        original_paths = sorted(original_dir.iterdir(), key=_key)
        gt_paths = sorted(gt_dir.iterdir(), key=_key)
        pairs.extend(zip(original_paths, gt_paths))
    return pairs


def dibco_path_pairs():
    root = Path(__file__).parent / 'datasets/dibco'
    if not root.exists() or not root.is_dir():
        raise Exception('Нужно сначала скачать и распаковать датасет DIBCO')
    return _collect_original_gt_pairs(root)


def phibd_path_pairs():
    root = Path(__file__).parent / 'datasets/phibd'
    if not root.exists() or not root.is_dir():
        raise Exception('Нужно сначала скачать и распаковать датасет PHIBD')
    return _collect_original_gt_pairs(root)


def custom_path_pairs():
    root = Path(__file__).parent / 'datasets/custom'
    if not root.exists() or not root.is_dir():
        warnings.warn('Нет папки custom')
        return []
    return _collect_original_gt_pairs(root)


class PairsGenerator(Sequence):
    """Вспомогательный класс для итерации по изображениям. Подходит для обучения моделей Keras.
    Нужен для того, чтобы не загружать весь датасет в память"""

    def __init__(self, batch_size, img_size, original_img_paths, gt_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.original_paths = original_img_paths
        self.gt_paths = gt_img_paths

    def __len__(self):
        """Количество батчей"""
        return len(self.original_paths) // self.batch_size

    def __getitem__(self, idx):
        """Возвращает батч (пару наборов изображений) по индексу"""
        i = idx * self.batch_size
        batch_original_img_paths = self.original_paths[i: i + self.batch_size]
        batch_gt_img_paths = self.gt_paths[i: i + self.batch_size]
        original = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_original_img_paths):
            img = load_img(path, target_size=self.img_size)
            original[j] = np.array(img) / 255
        gt = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_gt_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            gt[j] = np.expand_dims(img, 2) / 255
        return original, gt
