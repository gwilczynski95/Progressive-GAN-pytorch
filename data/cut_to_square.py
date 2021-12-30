import os
import shutil

from PIL import Image
import PIL
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import utils


def show_boundary_box_cut():
    # image_dir = 'C:/Users/Olaf1/Desktop/images'
    main_input_path = '/home/grzegorz/projects/museum/data/images'
    main_properly_cut_samples_path = '/home/grzegorz/projects/museum/data/properly_cut_images'
    main_badly_cut_samples_path = '/home/grzegorz/projects/museum/data/badly_cut_images'

    for category in os.listdir(main_input_path):
        if category == 'portrait':
            continue

        properly_cut_samples_path = os.path.join(main_properly_cut_samples_path, category)
        badly_cut_samples_path = os.path.join(main_badly_cut_samples_path, category)
        image_dir = os.path.join(main_input_path, category)

        try:
            os.mkdir(properly_cut_samples_path)
            os.mkdir(badly_cut_samples_path)
            properly_cut_samples = []
            badly_cut_samples = []
        except FileExistsError:
            properly_cut_samples = os.listdir(properly_cut_samples_path)
            badly_cut_samples = os.listdir(badly_cut_samples_path)

        images_list = os.listdir(image_dir)

        if len(properly_cut_samples) + len(badly_cut_samples) == len(images_list):
            continue

        for image_filename in images_list:
            if image_filename in properly_cut_samples or image_filename in badly_cut_samples:
                continue
            print(f'processing {category}: {image_filename}')
            image_path = os.path.join(image_dir, image_filename)
            proper_out_path = os.path.join(properly_cut_samples_path, image_filename)
            bad_out_path = os.path.join(badly_cut_samples_path, image_filename)
            try:
                raw_image = utils.load_image(image_path)
            except OSError:
                shutil.copy(image_path, bad_out_path)
                continue

            if raw_image is None or (not raw_image.shape[0]):
                shutil.copy(image_path, bad_out_path)
                continue

            if raw_image.shape[0] == raw_image.shape[1]:
                utils.save_image(proper_out_path, raw_image)
                continue

            # make the input image to be processible by SIFT
            grayscale_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
            stride_axis = np.argmax(grayscale_image.shape)

            window_size = grayscale_image.shape[1] if stride_axis == 0 else grayscale_image.shape[0]

            # calculate keypoints
            sift = cv2.xfeatures2d.SIFT_create()
            keypoints, descriptors = sift.detectAndCompute(raw_image, None)

            keypoints_magnitudes = set((keypoint.pt[int(not stride_axis)], keypoint.response) for keypoint in keypoints)
            window_scores = {'start': [], 'keypoints_quantity': [], 'keypoints_total_magnitude': []}
            axis_size = grayscale_image.shape[stride_axis]
            right_boundary = axis_size - window_size
            left_sided_windows = list(range(0, right_boundary, axis_size // 50))
            right_sided_windows = list(range(right_boundary, 0, -axis_size // 50))

            # calculate scores for each window
            for windows in (left_sided_windows, right_sided_windows):
                for window_start in windows:
                    window = (window_start, window_start + window_size)
                    window_magnitudes = [keypoint_magnitude[1] for keypoint_magnitude in keypoints_magnitudes if
                                         window[1] >= keypoint_magnitude[0] >= window[0]]
                    window_scores['start'].append(window_start)
                    window_scores['keypoints_quantity'].append(len(window_magnitudes))
                    window_scores['keypoints_total_magnitude'].append(sum(np.array(window_magnitudes)))

            scores = pd.DataFrame(window_scores)
            scores.sort_values(by=['start'], inplace=True)

            # create boundary box
            max_score_window_start = scores.start.loc[scores.keypoints_total_magnitude.idxmax()]
            first_point = (max_score_window_start, 0)
            second_point = (max_score_window_start + window_size, window_size)
            if not stride_axis:
                first_point = first_point[::-1]
                second_point = second_point[::-1]

            cut_image = raw_image[first_point[1]:second_point[1], first_point[0]:second_point[0]]
            # visualize(cut_image, first_point, keypoints, raw_image, second_point)
            utils.save_image(proper_out_path, cut_image)


def visualize(cut_image, first_point, keypoints, raw_image, second_point):
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(get_sift_decision_on_image(first_point, keypoints, raw_image, second_point))
    axs[1].imshow(cut_image)


def get_sift_decision_on_image(first_point, keypoints, raw_image, second_point):
    processed_image = cv2.drawKeypoints(raw_image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    processed_image = cv2.rectangle(processed_image, first_point, second_point, (255, 0, 0), 2)
    return processed_image


if __name__ == '__main__':
    show_boundary_box_cut()
