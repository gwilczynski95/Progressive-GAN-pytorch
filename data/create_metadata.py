import os

import pandas as pd

from data import utils

if __name__ == '__main__':
    path_to_data = '/home/grzegorz/projects/museum/data/properly_cut_images'
    categories = os.listdir(path_to_data)
    metadata = {
        'filename': [],
        'category': [],
        'size': []
    }
    for category in categories:
        category_path = os.path.join(path_to_data, category)
        image_names = [x for x in os.listdir(category_path) if '.jpg' in x]

        for image_name in image_names:
            image_path = os.path.join(category_path, image_name)

            image = utils.load_image(image_path)
            im_size = image.shape[0]

            metadata['filename'].append(image_name)
            metadata['category'].append(category)
            metadata['size'].append(im_size)

    out_df = pd.DataFrame.from_dict(metadata)
    out_df.to_csv(os.path.join(path_to_data, 'data_info.csv'))
