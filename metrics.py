from collections import OrderedDict

import numpy as np
from skimage.metrics import (
    mean_squared_error as mse,
    structural_similarity as _ssim,
)
from sklearn.metrics import jaccard_score

from imsplit import imsplit

def _prepare(img, invert=False):
    if not isinstance(img, np.ndarray):
        img = np.asarray(img)
    if img.max() > 1.1:
        img = img / 255
    if invert:
        img = 1 - img
    return img.astype(np.float)


def cpm(computed: np.ndarray, expected: np.ndarray, segment_size: int = 30, invert=True):
    """Count Pseudometric Measure

    Мера близости бинаризованных изображений на основе количества чёрных пикселей.
    Источник http://www.isa.ru/proceedings/images/documents/2013-63-3/t-3-13_85-94.pdf.
    """
    computed = _prepare(computed, invert)
    expected = _prepare(expected, invert)
    real_list = map(lambda p: p[0], imsplit(computed, segment_size, full_cover=True))
    expected_list = map(lambda p: p[0], imsplit(expected, segment_size, full_cover=True))

    result_sum = sum(abs(a.sum() - b.sum()) for a, b in zip(real_list, expected_list))
    return result_sum / expected.sum()


def ssim(computed, expected, invert=True):
    """structural similarity

    Мера близости бинаризованных изображений
    """
    computed = _prepare(computed, invert)
    expected = _prepare(expected, invert)
    return _ssim(computed, expected, data_range=1.0)


def iou(computed, expected, invert=True):
    computed = _prepare(computed, invert)
    expected = _prepare(expected, invert)
    return jaccard_score(computed, expected)


def get_metrics(computed: np.ndarray, expected: np.ndarray):
    return OrderedDict([
        ('mse', mse(computed, expected)),
        ('ssim', ssim(computed, expected)),
        ('cpm', cpm(computed, expected)),
        ('iou', iou(computed, expected)),
    ])


def print_metrics(computed: np.ndarray, expected: np.ndarray):
    print(f'MSE = {mse(computed, expected)}')
    print(f'SSIM = {ssim(computed, expected)}')
    print(f'CPM = {cpm(computed, expected)}')
    print(f'IOU = {iou(computed, expected)}')
    print(f'expected.max() == {expected.max()}')
    print(f'computed.max() == {computed.max()}')
