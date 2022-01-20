import os

import numpy as np
import torch
from pytorch_fid.inception import InceptionV3
from torchvision.transforms import transforms
from tqdm import tqdm

from conditional_proper_wikiart import WikiArtDataset
from fid.musem_fid import calculate_activation_statistics


def get_original_statistics(im_size, npz_ident, npz_path):
    npz_filename = f'{im_size}_{npz_ident}'
    absolute_npz_path = os.path.join(npz_path, npz_filename)
    with np.load(absolute_npz_path) as data:
        original_m = data['original_m']
        original_s = data['original_s']
    return original_m, original_s


if __name__ == '__main__':
    output_statistics_path = 'conditional_random_wikiart.npz'
    num_of_data_samples_for_fid = 10000
    inception_batch_size = 50
    inception_dims = 2048
    torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inception_num_workers = 8
    gen_batch_size = 100

    sizes = [4, 8, 16, 32, 64, 128, 256, 512]

    # get inception model
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[inception_dims]
    model = InceptionV3([block_idx]).to(torch_device)

    for size in sizes:
        print(f'size: {size}')
        # get FID statistics for dataset
        transform = transforms.Compose([
            transforms.Resize((size, size)),
        ])

        wikidataset = WikiArtDataset(
            '/home/grzegorz/projects/museum/data/properly_cut_images', current_size=size, transform=transform
        )
        wikiart_metadata = wikidataset.metadata[wikidataset.metadata['size'] >= wikidataset.current_size]

        original_data = np.zeros((num_of_data_samples_for_fid, size, size, 3), dtype=np.uint8)
        random_idxs = np.arange(num_of_data_samples_for_fid)
        np.random.shuffle(random_idxs)
        print('prepare data')
        categories = list(wikidataset.category_to_idx.keys())
        used_data = {key: [] for key in categories}
        for i in tqdm(range(num_of_data_samples_for_fid)):
            iloc_idx = 0
            while 1:
                random_category = categories[np.random.randint(0, len(categories))]
                category_df = wikiart_metadata[wikiart_metadata.category == random_category]
                random_data = category_df.iloc[np.random.randint(0, len(category_df)), :]
                if random_data['Unnamed: 0'] not in used_data[random_category]:
                    used_data[random_category].append(random_data['Unnamed: 0'])
                    iloc_idx = wikiart_metadata.index.get_loc(random_data['Unnamed: 0'])
                    break
            original_data[i, :, :, :] = wikidataset[iloc_idx][0]

        print('calculate statistics')
        m, s = calculate_activation_statistics(original_data, model, inception_batch_size,
                                                                 inception_dims, torch_device, inception_num_workers)

        # save this data
        np.savez(f'{size}_{output_statistics_path}', original_m=m, original_s=s)
