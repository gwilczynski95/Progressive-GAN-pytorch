import os
from abc import ABC

import numpy as np
import pandas as pd
import torch
from PIL import Image
from pytorch_fid.inception import InceptionV3
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from tqdm import tqdm

from data import utils as data_utils


class WikiArtDataset(Dataset, ABC):
    def __init__(self, root_dir, current_size=4, transform=None):
        self.metadata = pd.read_csv(os.path.join(root_dir, 'data_info.csv'))
        self.category_to_idx = {key: idx for idx, key in enumerate(self.metadata['category'].unique())}
        self.root_dir = root_dir
        self.transform = transform
        self.current_size = current_size

    def __len__(self):
        return len(self.metadata[self.metadata['size'] >= self.current_size])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample_data = self.metadata[self.metadata['size'] >= self.current_size].iloc[idx, :]
        img_path = os.path.join(
            self.root_dir,
            sample_data['category'],
            sample_data['filename']
        )
        image = data_utils.load_image(img_path)
        # must convert to PIL Image
        image = Image.fromarray(image, mode='RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.category_to_idx[sample_data['category']]


class MyDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]

        if self.transform:
            if x.dtype == np.float32:
                x = np.tanh(x) + 1
                x *= 127.5
                x = x.astype(np.uint8)
            if x.shape[0] == 3:
                x = np.transpose(x, (1, 2, 0))
            x = Image.fromarray(x).convert('RGB')
            x = self.transform(x)

        return x

    def __len__(self):
        return len(self.data)


def imagefolder_loader(path, data_batch_size):
    def loader(current_size, transform):
        data = WikiArtDataset(path, current_size=current_size, transform=transform)
        dataloader = DataLoader(data, shuffle=True, batch_size=data_batch_size, num_workers=0)  # todo: debug
        return dataloader
    return loader


def wikiart_sample_data(dataloader, image_size=4):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    loader = dataloader(image_size, transform)

    return loader


def get_activations(data, model, batch_size=50, dims=2048, device='cpu', num_workers=8):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- files       : ndarray of data
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = MyDataset(data, transform=transform)
    num_workers = 0 # todo: remove
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=num_workers)

    pred_arr = np.empty((len(data), dims))

    start_idx = 0

    for batch in tqdm(dataloader):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(data, model, batch_size=50, dims=2048,
                                    device='cpu', num_workers=8):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : ndarray of data
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(data, model, batch_size, dims, device, num_workers)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_fid_given_data(data_1, data_2, batch_size=50, device='cpu', dims=2048, num_workers=8):
    """Calculates the FID of two data sources"""
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx]).to(device)

    m1, s1 = calculate_activation_statistics(data_1, model, batch_size,
                                             dims, device, num_workers)
    m2, s2 = calculate_activation_statistics(data_2, model, batch_size,
                                             dims, device, num_workers)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value


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
        original_m, original_s = calculate_activation_statistics(original_data, model, inception_batch_size,
                                                                 inception_dims, torch_device, inception_num_workers)

        # save this data
        np.savez(f'{size}_{output_statistics_path}', original_m=original_m, original_s=original_s)
