from typing import Any

import torch


class AdaptiveAugment:
    def __init__(
            self, prev_ada_p: float = 0., ada_target: float = 0.6, ada_length: int = 500000, batch_size: int = 4,
            device: Any = 'cpu'
    ):
        """
        Calculates probability p of Adaptive Discriminator Augmentation.
        Class taken from: github.com/POSTECH-CVLab/PyTorch-StudioGAN

        :param prev_ada_p: if training is continued
        :param ada_target: what r_t value should be held  # FIXME CHYBA
        :param ada_length: how many images should be seen for p to come from 0 to 1# FIXME CHYBA
        :param batch_size:
        :param device: torch.device or string
        """
        self.prev_ada_p = prev_ada_p
        self.ada_target = ada_target
        self.ada_length = ada_length
        self.batch_size = batch_size
        self.rank = device

        self.ada_aug_step = 1. / (self.ada_length / self.batch_size)

    def initialize(self):
        self.ada_augment = torch.tensor([0.0, 0.0], device=self.rank)
        if self.prev_ada_p is not None:
            self.ada_aug_p = self.prev_ada_p
        else:
            self.ada_aug_p = 0.0
        return self.ada_aug_p

    def update(self, logits: torch.Tensor) -> float:
        ada_aug_data = torch.tensor((torch.sign(logits).sum().item(), logits.shape[0]), device=self.rank)
        self.ada_augment += ada_aug_data
        if self.ada_augment[1] > (self.batch_size * 4 - 1):
            authen_out_signs, num_outputs = self.ada_augment.tolist()
            r_t_stat = authen_out_signs / num_outputs
            sign = 1 if r_t_stat > self.ada_target else -1
            self.ada_aug_p += sign * self.ada_aug_step * num_outputs
            self.ada_aug_p = min(1.0, max(0.0, self.ada_aug_p))
            self.ada_augment.mul_(0.0)
        return self.ada_aug_p

    def set_batch_size(self, batch_size: int):
        self.batch_size = batch_size
        self.ada_aug_step = 1. / (self.ada_length / self.batch_size)
