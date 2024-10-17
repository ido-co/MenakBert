import os
import pickle
from pathlib import Path
from typing import List, Dict

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from consts import DATA_MODULES_CACHE
from dataset import textDataset


class HebrewDataModule(LightningDataModule):

    def __init__(
            self,
            train_paths,
            val_path,
            model,
            max_seq_length: int,
            min_seq_length: int,
            train_batch_size: int,
            val_batch_size: int,
            split_sentence,
            name: str,
            test_paths=None,
            **kwargs,
    ):
        super().__init__()
        self.model = model
        self.train_paths = train_paths
        self.val_paths = val_path
        self.test_paths = test_paths
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.min_seq_length = min_seq_length
        self.val_batch_size = val_batch_size
        self.split_sentence = split_sentence
        self.name = name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model, use_fast=True)

    def setup(self, stage: str = None):
        self.train_data = textDataset(
            self.train_paths,
            self.max_seq_length,
            self.min_seq_length,
            self.tokenizer,
            self.split_sentence
        )

        self.val_data = textDataset(
            self.val_paths,
            self.max_seq_length,
            self.min_seq_length,
            self.tokenizer,
            self.split_sentence
        )

        # TODO ask why it is set like this..I would check if the list is not empty
        if self.test_paths[0]:
            self.test_data = textDataset(
                self.test_paths,
                self.max_seq_length,
                self.min_seq_length,
                self.tokenizer,
                self.split_sentence
            )

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.train_batch_size, num_workers=12, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.val_batch_size, num_workers=12)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.val_batch_size, num_workers=12)


CACHE_DIR = Path("data_module_cache")


def save_to_cache(data_modules: List[HebrewDataModule], cache_file: str):
    """Saves the data modules to a cache file."""
    with open(CACHE_DIR / cache_file, 'wb') as f:
        pickle.dump(data_modules, f)


def load_from_cache(cache_file: str) -> List[HebrewDataModule]:
    """Loads the data modules from a cache file."""
    with open(CACHE_DIR / cache_file, 'rb') as f:
        return pickle.load(f)


def create_data_modules(params: Dict, use_cache: bool = False, force_refresh: bool = False) -> List[HebrewDataModule]:
    """Creates or loads data modules, with the option to use cache or force refresh."""

    # Define a unique cache filename based on parameters (optional)
    cache_file = DATA_MODULES_CACHE

    # Check if the cache directory exists, and create it if it doesn't
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # If using cache and cache file exists, and refresh is not forced, load from cache
    if use_cache and (CACHE_DIR / cache_file).exists() and not force_refresh:
        print("Loading data modules from cache...")
        return load_from_cache(cache_file)

    # Otherwise, create the data modules
    base_path = params['train_data']
    dirs = ['religion', 'pre_modern', 'early_modern', 'modern']
    testpath = [None, None, None, params['test_data']]

    data_modules = []
    for i, directory in enumerate(dirs):
        train_path = os.path.join(base_path, directory)

        dm = HebrewDataModule(
            train_paths=[train_path],
            val_path=[params['val_data']],
            model=params['model'],
            max_seq_length=params['maxlen'],
            min_seq_length=params['minlen'],
            train_batch_size=params['train_batch_size'],
            val_batch_size=params['val_batch_size'],
            split_sentence=params['split_sentence'],
            test_paths=[testpath[i]],
            name=directory
        )
        dm.setup()  # Ensure the data module is ready

        # Add the data module to the list
        data_modules.append(dm)

    # Save the newly created data modules to cache if caching is enabled
    if use_cache:
        print("Saving data modules to cache...")
        save_to_cache(data_modules, cache_file)

    return data_modules

# if __name__ == '__main__':
#     dm = HebrewDataModule(BACKBONE)
#     dm.prepare_data()
#     for batch in dm.train_dataloader():
#         break
