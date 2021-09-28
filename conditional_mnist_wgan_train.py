import json
import os

import numpy as np
from tqdm import tqdm

import torch
from torch import optim
from torch.autograd import grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from fid.load_mnist_model_and_fid_it import get_checkpoint_step_idx, load_config
from mnist_pggan import ConditionalGenerator, ConditionalDiscriminatorWgangp


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def imagefolder_loader(path, data_batch_size):
    def loader(transform):
        # data = datasets.ImageFolder(path, transform=transform)
        data = datasets.MNIST(path, transform=transform, download=True)
        data_loader = DataLoader(data, shuffle=True, batch_size=data_batch_size,
                                 # num_workers=4)
                                 num_workers=0)  # workaround for debugging
        return data_loader

    return loader


def mnist_sample_data(dataloader, image_size=4):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])

    loader = dataloader(transform)

    return loader


def train(
        generator, discriminator, g_running, loader, config, continue_training=False
):
    # fixme: right now n_critic is hardcoded
    n_critic = 1
    disc_loss_val = 0
    gen_loss_val = 0
    grad_loss_val = 0

    from datetime import datetime
    import os
    if continue_training:
        pbar = tqdm(range(config['current_overal_iteration'], config['current_overal_iteration'] + config['additional_iterations']))

        post_fix = '_'.join(config['model_folder_name'].split('_')[-3:])
        log_folder = config['model_folder_name']
        log_file_name = os.path.join(log_folder, 'train_log_' + post_fix)

        # setup iteration
        num_of_iterations_per_step = config['total_iter'] // config['max_step']
        # alpha = min(1,
        #             (2 / num_of_iterations_per_step) * (config['current_overal_iteration'] % num_of_iterations_per_step))
        step = int(config['current_overal_iteration'] / num_of_iterations_per_step) + 1
        if step > config['max_step']:
            # alpha = 1
            step = config['max_step']
        iteration = max(0, config['current_overal_iteration'] - num_of_iterations_per_step * (step - 1))

        # prepare dataset
        data_loader = mnist_sample_data(loader, 4 * 2 ** step)
        dataset = iter(data_loader)
        iterations_since_restart = 0
    else:
        step = config['init_step']  # can be 1 = 8, 2 = 16, 3 = 32, 4 = 64, 5 = 128, 6 = 128
        data_loader = mnist_sample_data(loader, 4 * 2 ** step)
        dataset = iter(data_loader)

        total_iter_remain = config['total_iter'] - (config['total_iter'] // config['max_step']) * (step - 1)

        last_step_additional_iterations = 100000

        pbar = tqdm(range(total_iter_remain + last_step_additional_iterations))

        date_time = datetime.now()
        post_fix = '%s_%s_%d_%d.txt' % (config['trial_name'], date_time.date(), date_time.hour, date_time.minute)
        log_folder = 'trial_%s_%s_%d_%d' % (config['trial_name'], date_time.date(), date_time.hour, date_time.minute)

        os.mkdir(log_folder)
        os.mkdir(log_folder + '/checkpoint')
        os.mkdir(log_folder + '/sample')

        config_info = {
            'generator': {
                'in_channel': generator.in_channel,
                'input_code_dim': generator.input_dim,
                'pixel_norm': generator.pixel_norm,
                'tanh': generator.tanh,
                'use_mnist_conv_blocks': generator.use_mnist_conv_blocks
            },
            'discriminator': {
                'feat_dim': discriminator.feat_dim,
                'use_mnist_conv_blocks': discriminator.use_mnist_conv_blocks
            },
            'batch_size': config['batch_size'],
            'learning_rate': config['learning_rate'],
            'total_iter': config['total_iter'],
            'max_step': config['max_step']
        }

        config_file_name = os.path.join(log_folder, 'train_config_' + post_fix[:-4] + '.json')
        if not os.path.exists(config_file_name):
            with open(config_file_name, 'w') as file:
                json.dump(config_info, file)

        log_file_name = os.path.join(log_folder, 'train_log_' + post_fix)
        if not os.path.exists(log_file_name):
            log_file = open(log_file_name, 'w')
            log_file.write('g,d,nll,onehot\n')
            log_file.close()

        # setup iteration
        iteration = 0

    # one = torch.FloatTensor([1]).to(device)
    one = torch.tensor(1, dtype=torch.float).to(config['device'])
    mone = one * -1

    for i in pbar:
        discriminator.zero_grad()

        if continue_training:
            iterations_since_restart += 1
        alpha = min(1, (2 / (config['total_iter'] // config['max_step'])) * iteration)
        if iteration != np.inf:
            if iteration > config['total_iter'] // config['max_step']:
                alpha = 0
                iteration = 0
                step += 1

                if step > config['max_step']:
                    iteration = np.inf
                    alpha = 1
                    step = config['max_step']
                data_loader = mnist_sample_data(loader, 4 * 2 ** step)
                dataset = iter(data_loader)

        try:
            real_image, label = next(dataset)

        except (OSError, StopIteration):
            dataset = iter(data_loader)
            real_image, label = next(dataset)

        if iteration != np.inf:
            iteration += 1

        ### 1. train Discriminator
        b_size = real_image.size(0)
        real_image = real_image.to(config['device'])
        label = label.to(config['device'])
        real_predict = discriminator(
            real_image, label, step=step, alpha=alpha)
        real_predict = real_predict.mean() \
                       - 0.001 * (real_predict ** 2).mean()  # why we do this here?
        real_predict.backward(mone)

        # sample input data: vector for Generator
        gen_z = torch.randn(b_size, config['generator']['input_code_dim']).to(config['device'])

        fake_image = generator(gen_z, label, step=step, alpha=alpha)
        fake_predict = discriminator(
            fake_image.detach(), label, step=step, alpha=alpha)
        fake_predict = fake_predict.mean()
        fake_predict.backward(one)

        ### gradient penalty for D
        eps = torch.rand(b_size, 1, 1, 1).to(config['device'])
        x_hat = eps * real_image.data + (1 - eps) * fake_image.detach().data
        x_hat.requires_grad = True
        hat_predict = discriminator(x_hat, label, step=step, alpha=alpha)
        grad_x_hat = grad(
            outputs=hat_predict.sum(), inputs=x_hat, create_graph=True)[0]
        grad_penalty = ((grad_x_hat.view(grad_x_hat.size(0), -1)
                         .norm(2, dim=1) - 1) ** 2).mean()
        grad_penalty = 10 * grad_penalty
        grad_penalty.backward()
        grad_loss_val += grad_penalty.item()
        disc_loss_val += (real_predict - fake_predict).item()

        config['d_optimizer'].step()

        ### 2. train Generator
        if (i + 1) % n_critic == 0:
            generator.zero_grad()
            discriminator.zero_grad()

            predict = discriminator(fake_image, label, step=step, alpha=alpha)

            loss = -predict.mean()
            gen_loss_val += loss.item()

            loss.backward()
            config['g_optimizer'].step()
            accumulate(g_running, generator)

        if (i + 1) % 1000 == 0 or i == 0:
            with torch.no_grad():
                gen_labels = np.ones((10, 10), dtype=np.int)
                for row_iter in range(10):
                    gen_labels[row_iter, :] *= row_iter
                gen_labels = torch.from_numpy(gen_labels.flatten()).to(config['device'])
                images = g_running(torch.randn(10 * 10, config['generator']['input_code_dim']).to(config['device']),
                                   gen_labels, step=step, alpha=alpha).data.cpu()

                utils.save_image(
                    images,
                    f'{log_folder}/sample/{str(i + 1).zfill(3)}.png',
                    nrow=10,
                    normalize=True,
                    range=(-1, 1))

        if (i + 1) % 2000 == 0 or i == 0:
            try:
                torch.save(g_running.state_dict(), f'{log_folder}/checkpoint/{str(i + 1).zfill(3)}_g.model')
                torch.save(discriminator.state_dict(), f'{log_folder}/checkpoint/{str(i + 1).zfill(3)}_d.model')
            except:
                pass

        if (i + 1) % 500 == 0:
            if continue_training:
                denominator_val = iterations_since_restart
                iterations_since_restart = 0
            else:
                denominator_val = 500

            state_msg = (f'{i + 1}; G: {gen_loss_val / (denominator_val // n_critic):.3f}; D: {disc_loss_val / denominator_val:.3f};'
                         f' Grad: {grad_loss_val / denominator_val:.3f}; Alpha: {alpha:.3f}')

            log_file = open(log_file_name, 'a+')
            new_line = "%.5f,%.5f\n" % (gen_loss_val / (denominator_val // n_critic), disc_loss_val / denominator_val)
            log_file.write(new_line)
            log_file.close()

            disc_loss_val = 0
            gen_loss_val = 0
            grad_loss_val = 0

            print(state_msg)
            # pbar.set_description(state_msg)


def prepare_training(resume_path=None):
    initial_step = 1
    if resume_path:
        checkpoints_path = os.path.join(resume_path, 'checkpoint')

        config = load_config(resume_path)

        generators_paths = [os.path.join(checkpoints_path, x) for x in os.listdir(checkpoints_path) if 'g' in x]
        generators_paths = sorted(generators_paths, key=get_checkpoint_step_idx)
        latest_generator_path = generators_paths[-1]
        discriminator_filename = latest_generator_path.split('/')[-1].split('_')[0] + '_d.model'
        latest_discriminator_path = os.path.join(checkpoints_path, discriminator_filename)
        config['model_folder_name'] = latest_generator_path.split('/')[-3]
        config['trial_name'] = config['model_folder_name'].split('_')[1]
        config['current_overal_iteration'] = get_checkpoint_step_idx(latest_generator_path) - 1
        config['init_step'] = initial_step
        config['additional_iterations'] = 100000  # fixme: here set how long the continuation has to go on
    else:
        trial_name = 'conditional_wgangp_test_1'
        input_code_size = 128
        channels = 64
        batch_size = 4
        pixel_norm = True
        tanh = False
        learning_rate = 0.001
        total_iterations = 90000
        maximal_step = 3
        use_mnist_conv_blocks = True
        config = {
            'generator': {
                'in_channel': channels,
                'input_code_dim': input_code_size,
                'pixel_norm': pixel_norm,
                'tanh': tanh,
                'use_mnist_conv_blocks': use_mnist_conv_blocks
            },
            'discriminator': {
                'feat_dim': channels,
                'use_mnist_conv_blocks': use_mnist_conv_blocks
            },
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'total_iter': total_iterations,
            'max_step': maximal_step,
            'trial_name': trial_name,
            'init_step': initial_step
        }

    config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    generator = ConditionalGenerator(**config['generator']).to(config['device'])
    discriminator = ConditionalDiscriminatorWgangp(**config['discriminator']).to(config['device'])
    g_running = ConditionalGenerator(**config['generator']).to(config['device'])

    # load checkpoint info
    if resume_path:
        generator.load_state_dict(torch.load(latest_generator_path))
        g_running.load_state_dict(torch.load(latest_generator_path))
        discriminator.load_state_dict(torch.load(latest_discriminator_path))

    g_running.train(False)

    config['g_optimizer'] = optim.Adam(generator.parameters(), lr=config['learning_rate'], betas=(0.0, 0.99))
    config['d_optimizer'] = optim.Adam(discriminator.parameters(), lr=config['learning_rate'], betas=(0.0, 0.99))

    accumulate(g_running, generator, 0)

    grzegos_data_path = '/home/grzegorz/grzegos_world/13_september_2021/mnist/'

    # loader = imagefolder_loader(args.path)
    loader = imagefolder_loader(grzegos_data_path, config['batch_size'])

    train(
        generator, discriminator, g_running, loader, config,
        continue_training=True if resume_path else False
    )


if __name__ == '__main__':
    path_to_continue_training = '/home/grzegorz/grzegos_world/13_september_2021/Progressive-GAN-pytorch/trial_test_5_2021-09-23_13_10'
    # prepare_training(path_to_continue_training)
    prepare_training()
