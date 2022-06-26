from pathlib import Path

import cv2 as cv

from imsplit import imsplit
from imcollect import dibco_path_pairs, custom_path_pairs


def make_segments(path_pairs, size: int, original_output_dir, gt_output_dir):
    original_output_dir = Path(original_output_dir)
    gt_output_dir = Path(gt_output_dir)
    counter = 0
    for num, (original, gt) in enumerate(path_pairs, 1):
        original = cv.imread(str(original), cv.IMREAD_COLOR)
        gt = cv.imread(str(gt), cv.IMREAD_COLOR)
        for (original_segment, _), (gt_segment, _) in zip(imsplit(original, size), imsplit(gt, size)):
            cv.imwrite(str(original_output_dir / f'{counter}.bmp'), original_segment)
            cv.imwrite(str(gt_output_dir / f'{counter}.bmp'), gt_segment)
            counter += 1
        print(f'{num} / {len(path_pairs)}, {counter} segments')


if __name__ == '__main__':
    original_dir = Path(__file__).parent / 'segments/original'
    gt_dir = Path(__file__).parent / 'segments/gt'
    # может быть ошибка, что папки уже существуют. Либо удали эти папки (скорее всего, это нужно),
    # либо добавь атрибут exist_ok=True
    original_dir.mkdir(parents=True)
    gt_dir.mkdir(parents=True)

    pairs = dibco_path_pairs()
    pairs.extend(custom_path_pairs())
    make_segments(pairs, 256, original_dir, gt_dir)
