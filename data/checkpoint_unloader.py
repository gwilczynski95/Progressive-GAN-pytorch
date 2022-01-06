import os
import zipfile
from pathlib import Path

if __name__ == '__main__':
    path_to_zips = '/home/grzegorz/projects/museum/trial_proper_conditional_wikiart_1_2022-01-03_16_22/checkpoint'
    destination_checkpoint_folder = '/home/grzegorz/projects/museum/trial_proper_conditional_wikiart_1_2022-01-03_16_22/checkpoint'

    zip_files = [x for x in os.listdir(path_to_zips) if '.zip' in x]
    for zip_filename in zip_files:
        zip_path = os.path.join(path_to_zips, zip_filename)
        with zipfile.ZipFile(zip_path, 'r') as file:
            file.extractall(destination_checkpoint_folder)
        os.remove(zip_path)

    # move it to
    checkpoints = Path(destination_checkpoint_folder).rglob('*.model')
    for checkpoint_path in checkpoints:
        filename = checkpoint_path.name
        source_filename = os.path.join(destination_checkpoint_folder, filename)
        os.rename(checkpoint_path, source_filename)

    # delete every file that is not a checkpoint
    not_checkpoints = [x for x in os.listdir(destination_checkpoint_folder) if '.model' not in x]
    for not_checkpoint_filename in not_checkpoints:
        not_checkpoint_path = os.path.join(destination_checkpoint_folder, not_checkpoint_filename)
        if os.path.isdir(not_checkpoint_path):
            os.rmdir(not_checkpoint_path)
        else:
            os.remove(not_checkpoint_path)
