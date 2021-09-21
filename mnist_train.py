import json

import numpy as np
from tqdm import tqdm

import torch
from torch import optim
from torch.autograd import grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from mnist_pggan import Generator, Discriminator


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


# def train(generator, discriminator, init_step, loader, total_iter=600000):
def train(generator, discriminator, init_step, loader, max_step, total_iter=300000):
    step = init_step  # can be 1 = 8, 2 = 16, 3 = 32, 4 = 64, 5 = 128, 6 = 128
    data_loader = mnist_sample_data(loader, 4 * 2 ** step)
    dataset = iter(data_loader)

    # todo: those comments were needed to be able to run this progan for mnist 32x32
    # total_iter = 600000
    # total_iter_remain = total_iter - (total_iter // 3) * (step - 1)
    total_iter_remain = total_iter - (total_iter // max_step) * (step - 1)

    last_step_additional_iterations = 100000

    pbar = tqdm(range(total_iter_remain + last_step_additional_iterations))

    disc_loss_val = 0
    gen_loss_val = 0
    grad_loss_val = 0

    from datetime import datetime
    import os
    date_time = datetime.now()
    post_fix = '%s_%s_%d_%d.txt' % (trial_name, date_time.date(), date_time.hour, date_time.minute)
    log_folder = 'trial_%s_%s_%d_%d' % (trial_name, date_time.date(), date_time.hour, date_time.minute)

    os.mkdir(log_folder)
    os.mkdir(log_folder + '/checkpoint')
    os.mkdir(log_folder + '/sample')

    config_info = {
        'generator': {
            'in_channel': generator.in_channel,
            'input_code_dim': generator.in_channel,
            'pixel_norm': generator.pixel_norm,
            'tanh': generator.tanh
        },
        'discriminator': {
            'feat_dim': discriminator.feat_dim
        },
        'batch_size': data_loader.batch_size,
        'learning_rate': g_optimizer.defaults['lr'],
        'total_iter': total_iter,
        'max_step': max_step
    }

    config_file_name = os.path.join(log_folder, 'train_config_' + post_fix[:-4] + '.json')
    with open(config_file_name, 'w') as file:
        json.dump(config_info, file)

    log_file_name = os.path.join(log_folder, 'train_log_' + post_fix)
    log_file = open(log_file_name, 'w')
    log_file.write('g,d,nll,onehot\n')
    log_file.close()

    from shutil import copy
    copy('train.py', log_folder + '/train_%s.py' % post_fix)
    copy('progan_modules.py', log_folder + '/model_%s.py' % post_fix)

    alpha = 0
    # one = torch.FloatTensor([1]).to(device)
    one = torch.tensor(1, dtype=torch.float).to(device)
    mone = one * -1
    iteration = 0

    for i in pbar:
        discriminator.zero_grad()

        alpha = min(1, (2 / (total_iter // max_step)) * iteration)
        if iteration != np.inf:
            if iteration > total_iter // max_step:
                alpha = 0
                iteration = 0
                step += 1

                if step > max_step:
                    iteration = np.inf
                    alpha = 1
                    step = max_step
                else:
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
        real_image = real_image.to(device)
        label = label.to(device)
        real_predict = discriminator(
            real_image, step=step, alpha=alpha)
        real_predict = real_predict.mean() \
                       - 0.001 * (real_predict ** 2).mean()  # why we do this here?
        real_predict.backward(mone)

        # sample input data: vector for Generator
        gen_z = torch.randn(b_size, input_code_size).to(device)

        fake_image = generator(gen_z, step=step, alpha=alpha)
        fake_predict = discriminator(
            fake_image.detach(), step=step, alpha=alpha)
        fake_predict = fake_predict.mean()
        fake_predict.backward(one)

        ### gradient penalty for D
        eps = torch.rand(b_size, 1, 1, 1).to(device)
        x_hat = eps * real_image.data + (1 - eps) * fake_image.detach().data
        x_hat.requires_grad = True
        hat_predict = discriminator(x_hat, step=step, alpha=alpha)
        grad_x_hat = grad(
            outputs=hat_predict.sum(), inputs=x_hat, create_graph=True)[0]
        grad_penalty = ((grad_x_hat.view(grad_x_hat.size(0), -1)
                         .norm(2, dim=1) - 1) ** 2).mean()
        grad_penalty = 10 * grad_penalty
        grad_penalty.backward()
        grad_loss_val += grad_penalty.item()
        disc_loss_val += (real_predict - fake_predict).item()

        d_optimizer.step()

        ### 2. train Generator
        if (i + 1) % n_critic == 0:
            generator.zero_grad()
            discriminator.zero_grad()

            predict = discriminator(fake_image, step=step, alpha=alpha)

            loss = -predict.mean()
            gen_loss_val += loss.item()

            loss.backward()
            g_optimizer.step()
            accumulate(g_running, generator)

        if (i + 1) % 1000 == 0 or i == 0:
            with torch.no_grad():
                images = g_running(torch.randn(5 * 10, input_code_size).to(device), step=step, alpha=alpha).data.cpu()

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
            state_msg = (f'{i + 1}; G: {gen_loss_val / (500 // n_critic):.3f}; D: {disc_loss_val / 500:.3f};'
                         f' Grad: {grad_loss_val / 500:.3f}; Alpha: {alpha:.3f}')

            log_file = open(log_file_name, 'a+')
            new_line = "%.5f,%.5f\n" % (gen_loss_val / (500 // n_critic), disc_loss_val / 500)
            log_file.write(new_line)
            log_file.close()

            disc_loss_val = 0
            gen_loss_val = 0
            grad_loss_val = 0

            print(state_msg)
            # pbar.set_description(state_msg)


if __name__ == '__main__':
    trial_name = 'test_1'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_code_size = 128
    channels = 128
    batch_size = 4
    n_critic = 1
    pixel_norm = True
    tanh = False
    learning_rate = 0.001
    initial_step = 1
    total_iterations = 90000
    maximal_step = 3

    generator = Generator(in_channel=channels, input_code_dim=input_code_size, pixel_norm=pixel_norm,
                          tanh=tanh).to(device)
    discriminator = Discriminator(feat_dim=channels).to(device)
    g_running = Generator(in_channel=channels, input_code_dim=input_code_size, pixel_norm=pixel_norm,
                          tanh=tanh).to(device)

    ## you can directly load a pretrained model here
    # generator.load_state_dict(torch.load('./tr checkpoint/150000_g.model'))
    # g_running.load_state_dict(torch.load('checkpoint/150000_g.model'))
    # discriminator.load_state_dict(torch.load('checkpoint/150000_d.model'))

    g_running.train(False)

    g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.0, 0.99))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.0, 0.99))

    accumulate(g_running, generator, 0)

    grzegos_data_path = '/home/grzegorz/grzegos_world/13_september_2021/mnist/'

    # loader = imagefolder_loader(args.path)
    loader = imagefolder_loader(grzegos_data_path, batch_size)

    train(generator, discriminator, initial_step, loader, maximal_step, total_iterations)
