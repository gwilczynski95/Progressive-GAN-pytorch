from typing import Any

import PIL
import cv2
import numpy as np
from PIL import Image
import pyvips


def load_image(image_path):
    image = pyvips.Image.new_from_file(image_path, access='sequential')
    mem_img = image.write_to_memory()
    try:
        raw_image = np.frombuffer(mem_img, dtype=np.uint8).reshape(image.height, image.width, 3)
    except ValueError:
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


if __name__ == '__main__':
    test_path = '/home/grzegorz/projects/museum/data/properly_cut_images/mythological-painting/32-Apollo and Marsyas.jpg'
    load_image(test_path)
