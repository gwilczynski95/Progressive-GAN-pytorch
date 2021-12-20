import copy
import os
import shutil

import numpy as np
import cv2
import matplotlib.pyplot as plt
import mtcnn
from PIL import Image


def main():
    image_dir = '/home/grzegorz/projects/museum/images/portrait'
    properly_cut_samples_path = ''
    ignored_samples_path = ''
    images_list = os.listdir(image_dir)

    dnn_model = mtcnn.MTCNN()
    for image_filename in images_list:
        image_path = os.path.join(image_dir, image_filename)
        raw_image = np.array(Image.open(image_path))
        if len(raw_image.shape) < 3:
            raw_image = cv2.imread(image_path)

        if raw_image.shape[0] == raw_image.shape[1]:
            pass  # save this image because it's already a rectangle

        # do it for dnn model
        dnn_faces = dnn_model.detect_faces(raw_image)
        for face_info in dnn_faces:
            x, y, width, height = face_info['box']
            image_with_rect = cv2.rectangle(raw_image, (x, y), (x + width, y + height), (0, 255, 0), 2)

        if not dnn_faces or len(dnn_faces) > 1:
            relative_image_path = os.path.join(ignored_samples_path, image_filename)
            shutil.copy(image_path, relative_image_path)
            continue

        x_center = (x + (x + width)) / 2
        y_center = (y + (y + height)) / 2

        cut_image = cut_based_on_point(raw_image, x_center, y_center)

        # save image
        out_path = os.path.join(properly_cut_samples_path, image_filename)
        cut_image = Image.fromarray(out_path)
        cut_image.save(out_path)

        # plt.imshow(raw_image)
        # plt.imshow(cut_image)

        # raw_image = cv2.circle(raw_image, (int(x_center), int(y_center)), radius=2, color=(0, 0, 255), thickness=1)

        # plt.imshow(image_with_rect)
        # plt.imshow(raw_image)

        # height = np.size(dnn_image,0)
        # width = np.size(dnn_image,1)
        # print(height,width)
        #
        # processed_image = cv2.rectangle(dnn_image, first_point, second_point, (255, 0, 0), 2)
        # plt.imshow(processed_image)


def fix_point_if_outside_boundary(point, min_bound, max_bound):
    return min(max_bound, max(min_bound, point))


def cut_based_on_point(raw_image, x_center, y_center):
    smaller_dim = np.argmin(raw_image.shape[:-1])
    bigger_dim = np.argmax(raw_image.shape[:-1])
    out_image_side_len = raw_image.shape[smaller_dim]

    min_boundary = raw_image.shape[smaller_dim] // 2
    max_boundary = raw_image.shape[bigger_dim] - min_boundary

    if smaller_dim:
        x_center = raw_image.shape[smaller_dim] // 2
        y_center = int(fix_point_if_outside_boundary(y_center, min_boundary, max_boundary))
    else:
        y_center = raw_image.shape[smaller_dim] // 2
        x_center = int(fix_point_if_outside_boundary(x_center, min_boundary, max_boundary))

    cut_image = raw_image[
                max(0, y_center - out_image_side_len // 2): min(raw_image.shape[0], y_center + out_image_side_len // 2),
                max(0, x_center - out_image_side_len // 2): min(raw_image.shape[1], x_center + out_image_side_len // 2),
                :
                ]
    # first_y_cut = y_center - raw_image.shape[smaller_dim] / 2
    # second_y_cut = y_center + raw_image.shape[smaller_dim] / 2
    # cut_image = raw_image[int(first_y_cut):int(second_y_cut), :int(raw_image.shape[1])]
    return cut_image


if __name__ == '__main__':
    main()
