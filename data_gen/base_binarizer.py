import json
import os
import random
import traceback
from functools import partial

import numpy as np
from tqdm import tqdm

import utils.commons.single_thread_env  # NOQA
from utils.commons.hparams import hparams

np.seterr(divide='ignore', invalid='ignore')


class BinarizationError(Exception):
    pass


class BaseBinarizer:
    def __init__(self, processed_data_dir=None):
        if processed_data_dir is None:
            processed_data_dir = hparams['processed_data_dir']
        self.processed_data_dir = processed_data_dir
        self.binarization_args = hparams['binarization_args']
        self.items = {}
        self.item_names = []

    def load_meta_data(self):
        processed_data_dir = self.processed_data_dir
        items_list = json.load(open(f"{processed_data_dir}/metadata.json"))
        for r in tqdm(items_list, desc='Loading meta data.'):
            item_name = r['item_name']
            self.items[item_name] = r
            self.item_names.append(item_name)
        if self.binarization_args['shuffle']:
            random.seed(1234)
            random.shuffle(self.item_names)

    @property
    def train_item_names(self):
        range_ = self._convert_range(self.binarization_args['train_range'])
        return self.item_names[range_[0]:range_[1]]

    @property
    def valid_item_names(self):
        range_ = self._convert_range(self.binarization_args['valid_range'])
        return self.item_names[range_[0]:range_[1]]

    @property
    def test_item_names(self):
        range_ = self._convert_range(self.binarization_args['test_range'])
        return self.item_names[range_[0]:range_[1]]

    def _convert_range(self, range_):
        if range_[1] == -1:
            range_[1] = len(self.item_names)
        return range_

    def meta_data(self, prefix):
        if prefix == 'valid':
            item_names = self.valid_item_names
        elif prefix == 'test':
            item_names = self.test_item_names
        else:
            item_names = self.train_item_names
        for item_name in item_names:
            yield self.items[item_name]

    def process(self):
        self.load_meta_data()
        os.makedirs(hparams['binary_data_dir'], exist_ok=True)
        self.process_data('valid')
        self.process_data('test')
        self.process_data('train')

    def process_data(self, prefix):
        raise NotImplementedError

    @classmethod
    def process_item(cls, item, binarization_args):
        raise NotImplementedError

    @property
    def num_workers(self):
        return int(os.getenv('N_PROC', hparams.get('N_PROC', os.cpu_count())))
