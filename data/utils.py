from typing import Any

import PIL
import cv2
import numpy as np
from PIL import Image


def load_image(image_path):
    raw_image = np.array(Image.open(image_path))
    if len(raw_image.shape) < 3:
        raw_image = cv2.imread(image_path)
    elif raw_image.shape[2] != 3:
        raw_image = np.array(Image.open(image_path).convert("RGB"))
    return raw_image


def save_image(path: str, data: Any) -> None:
    if not isinstance(data, PIL.Image.Image):
        data = PIL.Image.fromarray(data)
    data.save(path)
    print(f'saved: {path}')  # todo: change it to logging
