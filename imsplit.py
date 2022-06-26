import numpy as np


def _range_borders(start, finish, distance, step=None, full_cover=True):
    if finish - start < distance:
        return []
    if step is None:
        step = distance
    pairs = []
    for start_border in range(start, finish, step):
        finish_border = start_border + distance
        if finish_border > finish:
            if full_cover:
                pairs.append((finish - distance, finish))
            return pairs
        pairs.append((finish - distance, finish))
    return pairs


def replace_segments(crops, new_segments):
    return [(segment, borders) for segment, (_, borders) in zip(new_segments, crops)]


def imsplit(image, size, step=None, full_cover=True):
    """Метод для разделения изображения на квадратные сегменты

    :param image: изображение
    :param size: размер стороны сегмента в пикселях
    :param step: смещение сегмента. Нужно, если требуется перекрытие сегментов. Если None, сегменты
      не будут перекрываться (кроме последних крайних, что определяется параметром full_cover)
    :param full_cover: нужно ли добавлять крайние сегменты, если при этом будет перекрытие с соседним
      Например, при размере изображения 100 x 100 и размере сегмента 30 x 30 либо будет перекрытие
      сегментов, либо крайние правые и нижние сегменты будут проигнорированы (full_cover=False)
    :return: массив пар (сегмент, границы). Границы в формате PIL: (left, top, right, bottom)
    """
    image = np.asarray(image)
    if step is None:
        step = size
    crops = []
    h = image.shape[0]
    w = image.shape[1]
    for top, bottom in _range_borders(0, h, size, step, full_cover=full_cover):
        for left, right in _range_borders(0, w, size, step, full_cover=full_cover):
            crops.append((image[top:bottom, left:right], (left, top, right, bottom)))
    return crops


def get_shape(crops):
    assert len(crops) > 0, 'пустой массив сегментов'
    max_right = 0
    max_bottom = 0
    segment_shape = None
    for segment, (_, _, right, bottom) in crops:
        if segment_shape is None:
            segment_shape = segment.shape
        assert segment.shape == segment_shape, 'все сегменты должны иметь одинаковый размер'
        if max_right < right:
            max_right = right
        if max_bottom < bottom:
            max_bottom = bottom
    if len(segment_shape) == 2:
        shape = max_bottom, max_right
    elif segment_shape[2] == 1:
        shape = max_bottom, max_right, 1
    elif segment_shape[2] == 3:
        shape = max_bottom, max_right, 3
    else:
        raise Exception('неправильный атрибут shape у сегментов', segment_shape)
    return shape


def imjoin_max(crops):
    shape = get_shape(crops)
    max_image = np.zeros(shape, dtype=np.float64)
    for segment, (left, top, right, bottom) in crops:
        max_image[top:bottom, left:right] = np.maximum(max_image[top:bottom, left:right], segment)
    return max_image


def imjoin_min(crops):
    shape = get_shape(crops)
    min_image = np.zeros(shape, dtype=np.float64)
    for segment, (left, top, right, bottom) in crops:
        min_image[top:bottom, left:right] = np.minimum(min_image[top:bottom, left:right], segment)
    return min_image


def imjoin_average(crops):
    shape = get_shape(crops)
    sum_image = np.zeros(shape, dtype=np.float64)
    count_image = np.zeros(shape, dtype=np.float64)
    for segment, (left, top, right, bottom) in crops:
        sum_image[top:bottom, left:right] += segment
        count_image[top:bottom, left:right] += 1
    return sum_image / count_image
