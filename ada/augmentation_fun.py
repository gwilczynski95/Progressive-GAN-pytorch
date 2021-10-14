import os

import torch
import torchvision
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from ada.augment import AugmentPipe

if __name__ == '__main__':
    path_to_data = '/home/grzegorz/grzegos_world/13_september_2021/celeba/img_align_celeba'
    images_filenames = os.listdir(path_to_data)
    rows = 5
    cols = 5
    how_many_images = rows * cols
    image_size = 150

    out_image = np.zeros((rows * image_size, cols * image_size, 3), dtype=np.float32)

    input_image = np.array(Image.open(os.path.join(path_to_data, images_filenames[0])))[-image_size:, :image_size, :]
    # dataset = torchvision.datasets.CelebA(path_to_data, download=True)
    input_image = input_image.astype(np.float32) / 127.5 - 1
    input_image = np.reshape(input_image, (1,) + input_image.shape)

    augmenter = AugmentPipe(xflip=1,
                            rotate90=1,
                            xint=1,

                            scale=1,
                            rotate=1,
                            aniso=1,
                            xfrac=1,

                            brightness=1,
                            contrast=1,
                            lumaflip=1,
                            hue=1,
                            saturation=1,

                            imgfilter=1,
                            noise=1,
                            cutout=1,
                            )
    temp_images = np.concatenate([input_image for _ in range(rows)], axis=0)
    tensor_images = torch.from_numpy(np.transpose(temp_images, (0, 3, 1, 2)))
    for col_idx in range(cols):
        p = 1. / cols * col_idx
        augmenter.p = torch.tensor(p)
        augmented_images = np.transpose(augmenter.forward(tensor_images).numpy(), (0, 2, 3, 1))
        for img_idx in range(augmented_images.shape[0]):
            temp_image = augmented_images[img_idx]
            temp_image = (temp_image - temp_image.min()) / (temp_image.max() - temp_image.min())
            out_image[
                img_idx * image_size: (img_idx + 1) * image_size,
                col_idx * image_size: (col_idx + 1) * image_size,
                :
            ] = temp_image

    stop = 1
