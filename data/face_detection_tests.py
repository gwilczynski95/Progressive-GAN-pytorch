import copy
import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
import mtcnn
from PIL import Image


def main():

    image_dir = 'C:/Users/Olaf1/Desktop/images'
    images_list = os.listdir(image_dir)

    dnn_model = mtcnn.MTCNN()
    patches = []
    for image_filename in images_list:
        image_filename = 'C:/Users/Olaf1/Desktop/images/' + image_filename
        raw_image = np.array(Image.open(image_filename))
        if len(raw_image.shape) < 3:
            raw_image = cv2.imread(image_filename)

        # do it for dnn model
        dnn_image = copy.deepcopy(raw_image)
        dnn_faces = dnn_model.detect_faces(dnn_image)
        for face_info in dnn_faces:
            x, y, width, height = face_info['box']
            dnn_image = cv2.rectangle(dnn_image, (x, y), (x + width, y + height), (0, 255, 0), 2)

        xCenter = (x + (x + width)) / 2
        yCenter = (y + (y + height)) / 2
        #TODO add that when first_y_cut is less than 0 then first is 0, second is width
        #TODO add when image is horizontal
        first_y_cut = yCenter - raw_image.shape[1]/2
        second_y_cut = yCenter + raw_image.shape[1] / 2

        cut_image = raw_image[int(first_y_cut):int(second_y_cut), 0:int(raw_image.shape[1])]
        plt.imshow(cut_image)


        dnn_image = cv2.circle(dnn_image, (int(xCenter), int(yCenter)), radius=2, color=(0, 0, 255), thickness=1)


        plt.imshow(dnn_image)


        #
        # height = np.size(dnn_image,0)
        # width = np.size(dnn_image,1)
        # print(height,width)
        #
        # processed_image = cv2.rectangle(dnn_image, first_point, second_point, (255, 0, 0), 2)
        # plt.imshow(processed_image)

if __name__ == '__main__':
    main()