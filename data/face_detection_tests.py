import copy
import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
import mtcnn
from PIL import Image


def main():
    classifier_main_path = '/home/grzegorz/grzegos_world/13_september_2021/Progressive-GAN-pytorch/data/haarcascades'
    images = [
        '/home/grzegorz/grzegos_world/11_july_2021/images/portrait/11868-The Sorcerer of Hiva Oa (Marquesan Man in the Red Cape).jpg',
        '/home/grzegorz/grzegos_world/11_july_2021/images/portrait/8592-Portrait of Jean Cepeinick.jpg',
        '/home/grzegorz/grzegos_world/11_july_2021/images/portrait/11712-Portrait of actress Fanny Janauscher.jpg',
        '/home/grzegorz/grzegos_world/11_july_2021/images/portrait/11815-Portrait of Tadeusz Błotnicki with Medusa.jpg',
        '/home/grzegorz/grzegos_world/11_july_2021/images/portrait/11814-The monk and the beggar.jpg',
        '/home/grzegorz/grzegos_world/11_july_2021/images/portrait/11851-George Sotter.jpg',
        '/home/grzegorz/grzegos_world/11_july_2021/images/portrait/3216-George Parker (1755–1842), 4th Earl of Macclesfield.jpg',
        '/home/grzegorz/grzegos_world/11_july_2021/images/portrait/3266-Archduke Ferdinand and Archduchess Maria Anna of Austria.jpg',
        '/home/grzegorz/grzegos_world/11_july_2021/images/portrait/7187-Madame Louis Fran&#231;ois Godinot, born Victoire Pauline Thiolliere de L&#39;Isle.jpg',
        '/home/grzegorz/grzegos_world/11_july_2021/images/portrait/8931-Little Girl with Blond Hair.jpg',
        '/home/grzegorz/grzegos_world/11_july_2021/images/portrait/8935-La Fille De L&#39;Emir.jpg',
        '/home/grzegorz/grzegos_world/11_july_2021/images/portrait/10430-Augustus II the Strong.jpg',
        '/home/grzegorz/grzegos_world/11_july_2021/images/portrait/11536-Boy.jpg',
        '/home/grzegorz/grzegos_world/11_july_2021/images/portrait/12818-Portret Dziecka.jpg',
        '/home/grzegorz/grzegos_world/11_july_2021/images/portrait/13171-Young Woman.jpg'
    ]
    classifiers = os.listdir(classifier_main_path)

    dnn_model = mtcnn.MTCNN()
    patches = []
    for image_path in images:

        raw_image = np.array(Image.open(image_path))
        if len(raw_image.shape) < 3:
            raw_image = cv2.imread(image_path)

        # do it for dnn model
        dnn_image = copy.deepcopy(raw_image)
        dnn_faces = dnn_model.detect_faces(dnn_image)
        for face_info in dnn_faces:
            x, y, width, height = face_info['box']
            dnn_image = cv2.rectangle(dnn_image, (x, y), (x + width, y + height), (0, 255, 0), 2)

        plt.imshow(dnn_image)


if __name__ == '__main__':
    main()
