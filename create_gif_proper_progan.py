import os

import PIL.Image
from PIL import Image, ImageDraw
import numpy as np
import imageio
from tqdm import tqdm
from pygifsicle import optimize

from fid.load_cifar_model_and_fid_it import load_config


def get_sample_iteration(checkpoint_path):
    return int(checkpoint_path.split('/')[-1].split('.')[0])


def load_image(path, dtype='uint8'):
    sample = Image.open(path)
    sample.load()
    return np.asarray(sample, dtype=dtype)


def get_step_and_alpha(config_info, sample_iteration_idx):
    # calculate current generator step (its state of progression) and alpha
    iterations_per_mini_step = config_info['images_seen_per_mini_step'] // config_info['batch_size']
    step = config_info['init_step']
    if step == 1:
        iter_count = sample_iteration_idx - iterations_per_mini_step
    else:
        iter_count = sample_iteration_idx - 2 * iterations_per_mini_step

    if iter_count <= 0:
        step_iteration = sample_iteration_idx
    else:
        while iter_count > 0:
            step += 1
            step_iteration = iter_count
            iter_count -= 2 * iterations_per_mini_step
            if step == config_info['max_step']:
                break

    alpha = min(1, step_iteration / iterations_per_mini_step)
    return step, alpha


def get_images_from_sample(data, im_size, im_rows, im_cols, padding_px):
    sample_images = []
    for row_idx in range(im_rows):
        start_row_idx = padding_px * (row_idx + 1) + (row_idx * im_size)
        stop_row_idx = padding_px * (row_idx + 1) + ((row_idx + 1) * im_size)
        for col_idx in range(im_cols):
            start_col_idx = padding_px * (col_idx + 1) + (col_idx * im_size)
            stop_col_idx = padding_px * (col_idx + 1) + ((col_idx + 1) * im_size)
            temp_image = data[start_row_idx:stop_row_idx, start_col_idx:stop_col_idx, :]
            sample_images.append(temp_image)
    return sample_images


def from_read_samples_generate_resized_output(samples, out_im_shape, im_rows, im_cols, padding_px, dtype):
    resized_sample_data = np.zeros([
        out_im_shape[0] * im_rows + (padding_px * (im_rows + 1)),
        out_im_shape[1] * im_cols + (padding_px * (im_cols + 1)),
        3
    ], dtype=dtype)

    for row_idx in range(im_rows):
        start_row_idx = padding_px * (row_idx + 1) + (row_idx * out_im_shape[0])
        stop_row_idx = padding_px * (row_idx + 1) + ((row_idx + 1) * out_im_shape[0])
        for col_idx in range(im_cols):
            start_col_idx = padding_px * (col_idx + 1) + (col_idx * out_im_shape[1])
            stop_col_idx = padding_px * (col_idx + 1) + ((col_idx + 1) * out_im_shape[1])
            index = im_cols * row_idx + col_idx
            resized_sample_data[
            start_row_idx: stop_row_idx, start_col_idx:stop_col_idx, :
            ] = resize_sample(samples[index], out_im_shape, dtype)
    return resized_sample_data


def get_info_on_image(resized_sample_data_shape, image_shape, step, alpha, dtype):
    alpha_size = 350
    text_height = 150
    info_image = np.zeros((resized_sample_data_shape[0], resized_sample_data_shape[0], 3), dtype=dtype)
    step_and_shape_text_image = get_step_and_shape_text_image(step, image_shape,
                                                         out_shape=(resized_sample_data_shape[0], text_height))
    info_image[
        100:100 + text_height, :, :
    ] = step_and_shape_text_image
    alpha_text_image = get_alpha_text_image((alpha_size, text_height))
    left_alpha_padding = (resized_sample_data_shape[0] - alpha_size) // 2
    info_image[
        475: 475 + text_height, left_alpha_padding: left_alpha_padding + alpha_size, :
    ] = alpha_text_image
    info_image[
        350: 450, 50: 600, :
    ] = get_progress_bar(alpha, output_shape=(100, 550, 3))
    return info_image


def get_step_and_shape_text_image(step, shape, out_shape=(650, 150), dtype='uint8'):
    img = Image.new('RGB', (115, 11))
    d = ImageDraw.Draw(img)
    d.text((0, 0), f'Step {step}, shape {shape}x{shape}', fill=(255, 255, 255))
    return np.asarray(img.resize(out_shape, resample=PIL.Image.NEAREST), dtype=dtype)


def get_alpha_text_image(out_shape=(150, 150), dtype='uint8'):
    img = Image.new('RGB', (30, 11))
    d = ImageDraw.Draw(img)
    d.text((0, 0), 'alpha', fill=(255, 255, 255))
    return np.asarray(img.resize(out_shape, resample=PIL.Image.NEAREST), dtype=dtype)


def get_progress_bar(alpha_value, output_shape=(100, 550, 3), dtype='uint8'):
    progress_bar = np.zeros(output_shape, dtype=dtype)

    # make left bracket
    progress_bar[:, :10, :] = 255
    progress_bar[:10, :30, :] = 255
    progress_bar[-10:, :30, :] = 255

    # make right bracket
    progress_bar[:, -10:, :] = 255
    progress_bar[:10, -30:, :] = 255
    progress_bar[-10:, -30:, :] = 255

    # make progress bar
    progress_bar_columns = int(alpha_value * (output_shape[1] - 40))
    progress_bar[20:80, 20: 20 + progress_bar_columns] = 255
    return progress_bar


def resize_sample(data, out_shape, dtype):
    im_data = Image.fromarray(data)
    resized_im_data = im_data.resize(out_shape, resample=PIL.Image.NEAREST)
    return np.asarray(resized_im_data, dtype=dtype)


def main():
    path_to_training_info = '/home/grzegorz/grzegos_world/14_november_2021/trial_proper_cifar_test_1_2021-10-01_19_54'
    samples_path = os.path.join(path_to_training_info, 'sample')
    samples_paths = sorted(os.listdir(samples_path), key=get_sample_iteration)
    samples_dtype = 'uint8'
    fps = 50

    config = load_config(path_to_training_info)

    num_of_image_rows = 5
    num_of_image_cols = 10
    output_images_shape = [100, 100]
    input_padding_in_px = 2
    output_padding_in_px = 25

    step_to_size = {
        1: 4,
        2: 8,
        3: 16,
        4: 32
    }

    with imageio.get_writer(os.path.join(path_to_training_info, 'samples_through_training.gif'), mode='I', format='GIF-PIL', quantizer=0) as writer:
        for sample_filename in tqdm(samples_paths):
            sample_iteration = int(sample_filename.split('.')[0]) - 1
            sample_data = load_image(os.path.join(samples_path, sample_filename), samples_dtype)

            sample_step, sample_alpha = get_step_and_alpha(config, sample_iteration)
            image_size = step_to_size[sample_step]

            all_sample_images = get_images_from_sample(
                sample_data, image_size, num_of_image_rows, num_of_image_cols, input_padding_in_px
            )

            resized_sample_data = from_read_samples_generate_resized_output(
                all_sample_images, output_images_shape, num_of_image_rows, num_of_image_cols, output_padding_in_px,
                samples_dtype
            )

            info_image = get_info_on_image(resized_sample_data.shape, image_size, sample_step, sample_alpha, samples_dtype)

            gif_sample = np.zeros((info_image.shape[0], resized_sample_data.shape[1] + info_image.shape[1], 3), dtype=samples_dtype)
            gif_sample[
                :, :info_image.shape[1], :
            ] = info_image
            gif_sample[
                :, info_image.shape[1]:, :
            ] = resized_sample_data
            writer.append_data(gif_sample)
    # optimize(os.path.join(path_to_training_info, 'samples_through_training.gif'))

if __name__ == '__main__':
    main()
