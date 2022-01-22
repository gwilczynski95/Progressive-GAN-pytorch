import copy
import os

import torch

from fid.load_cifar_model_and_fid_it import get_checkpoint_step_idx, load_config
from progan_modules import ConditionalCorrectGenerator, ConditionalCorrectDiscriminatorWgangp, \
    ConditionalCorrectGenerator512, ConditionalCorrectDiscriminatorWgangp512


def accumulate_generator(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par2.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def accumulate_discriminator(bigger, smaller, decay=0.999):
    bigger_data = dict(bigger.named_parameters())
    smaller_data = dict(smaller.named_parameters())

    bigger_layer_names = list(bigger_data.keys())
    smaller_layer_names = list(smaller_data.keys())

    layer_categories = list(set(map(lambda x: x.split('.')[0], smaller_layer_names)))

    for category in layer_categories:
        bigger_category_layer_names = list(filter(lambda x: x.split('.')[0] == category, bigger_layer_names))
        smaller_category_layer_names = list(filter(lambda x: x.split('.')[0] == category, smaller_layer_names))

        for idx in range(1, len(smaller_category_layer_names) + 1):
            bigger_layer_name = bigger_category_layer_names[-idx]
            smaller_layer_name = smaller_category_layer_names[-idx]

            bigger_data[bigger_layer_name].data.mul_(decay).add_(1 - decay, smaller_data[smaller_layer_name])


if __name__ == '__main__':
    path_to_smaller_checkpoints = '/home/grzegorz/projects/museum/trial_proper_conditional_wikiart_1_2022-01-03_16_22/checkpoint'
    smaller_config = load_config('/home/grzegorz/projects/museum/trial_proper_conditional_wikiart_1_2022-01-03_16_22')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # prepare bigger config
    bigger_config = copy.deepcopy(smaller_config)
    bigger_config['generator']['max_step'] = 8

    generators_paths = [
        os.path.join(path_to_smaller_checkpoints, x) for x in os.listdir(path_to_smaller_checkpoints) if '_g' in x
    ]
    discriminator_paths = [
        os.path.join(path_to_smaller_checkpoints, x) for x in os.listdir(path_to_smaller_checkpoints) if '_d' in x
    ]
    generators_paths = sorted(generators_paths, key=get_checkpoint_step_idx)
    discriminator_paths = sorted(discriminator_paths, key=get_checkpoint_step_idx)

    # load smaller nets
    generator_checkpoint_path = generators_paths[-1]
    discriminator_checkpoint_path = discriminator_paths[-1]

    generator_smaller = ConditionalCorrectGenerator(**smaller_config['generator']).to(device)
    generator_smaller.load_state_dict(torch.load(generator_checkpoint_path, map_location=device))
    discriminator_smaller = ConditionalCorrectDiscriminatorWgangp(**smaller_config['discriminator']).to(device)
    discriminator_smaller.load_state_dict(torch.load(discriminator_checkpoint_path, map_location=device))

    generator_bigger = ConditionalCorrectGenerator512(**bigger_config['generator']).to(device)
    discriminator_bigger = ConditionalCorrectDiscriminatorWgangp512(**smaller_config['discriminator']).to(device)

    # copy params to bigger net
    # G
    accumulate_generator(generator_bigger, generator_smaller, decay=0)
    # del generator_smaller

    # D
    accumulate_discriminator(discriminator_bigger, discriminator_smaller, decay=0)
    # del discriminator_smaller

    # test if outputs are the same
    gen_z = torch.randn(1, bigger_config['generator']['input_code_dim']).to(device)
    import numpy as np
    label = torch.from_numpy(np.array([1])).to(device)

    fake_image = generator_bigger(gen_z, label, step=6, alpha=1.)
    fake_image_1 = generator_smaller(gen_z, label, step=6, alpha=1.)
    fake_predict = discriminator_bigger(
        fake_image.detach(), label, step=6, alpha=1.)
    fake_predict_1 = discriminator_smaller(
        fake_image.detach(), label, step=6, alpha=1.)

    assert (fake_image == fake_image_1).numpy().min
    assert (fake_predict == fake_predict_1).numpy().min
