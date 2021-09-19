import os
import json

import torch
import numpy as np
from numpy.random import default_rng
from pytorch_fid.inception import InceptionV3
from torchvision import datasets
from tqdm import tqdm

from mnist_pggan import Generator
from musem_fid import calculate_activation_statistics, calculate_frechet_distance


def imagefolder_loader(path):
    loaded_data = datasets.MNIST(path, download=True)
    return loaded_data.train_data.numpy()


def get_mnist_data(path_to_data, no_of_data):
    data = imagefolder_loader(path_to_data)
    rng = default_rng()
    indexes = rng.choice(np.arange(len(data)), size=no_of_data, replace=False)
    return data[indexes]


def get_checkpoint_step_idx(checkpoint_path):
    return int(checkpoint_path.split('/')[-1].split('_')[0])


def load_config(path):
    config_file_path = [x for x in os.listdir(path) if 'config' in x][0]
    with open(os.path.join(path, config_file_path), 'r') as file:
        conf = json.load(file)
    return conf


def load_prev_fid_statistics(path):
    try:
        with open(path, 'r') as file:
            fid_out = json.load(file)
            prev_checkpoint = max([int(x) for x in fid_out.keys()])
    except FileNotFoundError:
        fid_out = {}
        prev_checkpoint = -1
    return fid_out, prev_checkpoint


if __name__ == '__main__':
    num_of_data_samples_for_fid = 5
    inception_batch_size = 50
    inception_dims = 2048
    torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inception_num_workers = 0

    path_to_training_info = '/trial_test1_2021-09-06_19_23'
    checkpoints_path = os.path.join(path_to_training_info, 'checkpoint')
    output_path_for_fid_score = os.path.join(path_to_training_info, 'fid_score.json')

    # load config
    config = load_config(path_to_training_info)

    # config['total_iter'] = 30000  # fixme: hotfix for debug purposes

    # load generator paths
    generators_paths = [os.path.join(checkpoints_path, x) for x in os.listdir(checkpoints_path) if 'g' in x]
    generators_paths = sorted(generators_paths, key=get_checkpoint_step_idx)

    # get inception model
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[inception_dims]
    model = InceptionV3([block_idx]).to(torch_device)

    # get FID statistics for dataset
    original_mnist_data = get_mnist_data('', num_of_data_samples_for_fid)
    original_m, original_s = calculate_activation_statistics(original_mnist_data, model, inception_batch_size,
                                                             inception_dims, torch_device, inception_num_workers)

    # load previous FID
    fid_output, previous_checkpoint = load_prev_fid_statistics(output_path_for_fid_score)

    # calculate generator checkpoints FID score
    for generator_path in generators_paths:
        current_checkpoint_iteration_idx = get_checkpoint_step_idx(generator_path) - 1

        # if FID score is calculated for this checkpoint
        if previous_checkpoint >= current_checkpoint_iteration_idx:
            continue

        # load generator model
        generator = Generator(**config['generator']).to(torch_device)
        generator.load_state_dict(torch.load(generator_path))

        # calculate current generator step (its state of progression) and alpha
        num_of_iterations_per_step = config['total_iter'] // config['max_step']
        alpha = min(1,
                    (2 / num_of_iterations_per_step) * (current_checkpoint_iteration_idx % num_of_iterations_per_step))
        step = int(current_checkpoint_iteration_idx / num_of_iterations_per_step) + 1
        if step > config['max_step']:
            alpha = 1
            step = config['max_step']

        # generate data
        all_gen_data = []
        with torch.no_grad():
            gen_batch_size = 100
            for idx in range(num_of_data_samples_for_fid // gen_batch_size):
                gen_z = torch.randn(gen_batch_size, config['generator']['input_code_dim']).to(torch_device)
                gen_data = generator(gen_z, step=step, alpha=alpha)
                gen_data = gen_data.squeeze(1).numpy()
                if isinstance(all_gen_data, list):
                    all_gen_data = gen_data
                else:
                    all_gen_data = np.vstack((all_gen_data, gen_data))

        # calculate FID score for generated data
        generated_m, generated_s = calculate_activation_statistics(
            all_gen_data, model, inception_batch_size,
            inception_dims, torch_device, inception_num_workers
        )
        current_fid_value = calculate_frechet_distance(
            original_m, original_s, generated_m, generated_s
        )
        print('----------------')
        print(f'Iteration: {current_checkpoint_iteration_idx}')
        print(f'Step: {step}')
        print(f'Alpha: {alpha}')
        print(f'FID value: {current_fid_value}')
        print('----------------')
        fid_output[current_checkpoint_iteration_idx] = current_fid_value
        previous_checkpoint = current_checkpoint_iteration_idx

        # save calculated FID score
        with open(output_path_for_fid_score, 'w') as file:
            json.dump(fid_output, file)
