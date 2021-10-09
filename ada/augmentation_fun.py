import torch
import torchvision
import numpy as np

from ada.augment import AugmentPipe

if __name__ == '__main__':
    path_to_data = '/home/grzegorz/grzegos_world/13_september_2021/cifar/'
    dataset = torchvision.datasets.CIFAR10(path_to_data, train=True, download=True)
    tensor_image = torch.from_numpy(np.transpose(dataset.data[:1].astype(np.float32), (0, 3, 1, 2)) / 127.5 - 1)
    stop = 1
    augmenter = AugmentPipe(xflip=1, rotate90=1, xint=1, xint_max=1,
                            scale=1, rotate=1, aniso=1, xfrac=1, scale_std=1, rotate_max=1, aniso_std=1,
                            xfrac_std=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1,
                            brightness_std=1, contrast_std=1, hue_max=1, saturation_std=1, imgfilter=1,
                            imgfilter_bands=[1, 1, 1, 1], imgfilter_std=1, noise=1, cutout=1, noise_std=1,
                            cutout_size=1
                            )
    _ = augmenter.forward(tensor_image)
    stop = 1
