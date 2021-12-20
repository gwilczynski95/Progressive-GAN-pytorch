import os
from PIL import Image
import PIL
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



def show_boundary_box_cut():
    image_dir = 'C:/Users/Olaf1/Desktop/images'
    images_list = os.listdir(image_dir)
    n = 0
    for image_filename in images_list:
        image_path = os.path.join(image_dir, image_filename)
        raw_image = cv2.imread(image_path)

        grayscale_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
        stride_axis = np.argmax(grayscale_image.shape)
        if grayscale_image.shape[0] == grayscale_image.shape[1]:
            # todo: this image is correct - pick it
            continue
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

        processed_image = cv2.drawKeypoints(raw_image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # create boundary box
        max_score_window_start = scores.start.loc[scores.keypoints_total_magnitude.idxmax()]
        first_point = (max_score_window_start, 0)
        second_point = (max_score_window_start + window_size, window_size)
        if not stride_axis:
            first_point = first_point[::-1]
            second_point = second_point[::-1]

        #cut image
        cut_image = raw_image[first_point[1]:second_point[1], first_point[0]:second_point[0]]
        #TODO give right pathway for save
        processed_image = cv2.rectangle(processed_image, first_point, second_point, (255, 0, 0), 2)
        plt.imshow(processed_image)
        n += 1
        name = f"image{n}.jpg"
        cv2.imwrite(name, cut_image)




if __name__ == '__main__':
    show_boundary_box_cut()
