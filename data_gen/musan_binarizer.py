import os
from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm

from utils.commons.hparams import hparams
from utils.commons.indexed_datasets import IndexedDatasetBuilder

class MUSANBinarizer:
    def process(self):
        os.makedirs(hparams['binary_data_dir'], exist_ok=True)
        wav_paths = []
        for root, dirs, files in os.walk(hparams['raw_data_dir']):
            if 'short-musan' in root:
                continue
            if len(files) > 0:
                for f_name in files:
                    if Path(f_name).suffix == '.wav':
                        wav_paths.append(os.path.join(root, f_name))

        musan_noise_dict = {}
        for wav_path in tqdm(wav_paths, total=len(wav_paths)):
            item_name = Path(wav_path).stem
            if item_name in musan_noise_dict:
                print(f'skip {wav_path}: duplicated item name')
                continue
            wav, _ = librosa.core.load(wav_path, sr=hparams['audio_sample_rate'])
            musan_noise_dict[item_name] = wav.astype(np.float16)

        item_names = list(musan_noise_dict.keys())
        np.random.shuffle(item_names)

        valid_ratio = hparams['valid_ratio']
        test_ratio = hparams['test_ratio']
        valid_ids = item_names[:int(len(item_names) * valid_ratio)]
        test_ids = item_names[int(len(item_names) * valid_ratio): int(len(item_names) * valid_ratio) + int(
            len(item_names) * test_ratio)]
        train_ids = item_names[int(len(item_names) * valid_ratio) + int(len(item_names) * test_ratio):]

        train_set = {k: musan_noise_dict[k] for k in train_ids}
        test_set = {k: musan_noise_dict[k] for k in test_ids}
        valid_set = {k: musan_noise_dict[k] for k in valid_ids}

        def save_feat(feat_dict, prefix, feat_dir):
            builder = IndexedDatasetBuilder(f"{feat_dir}/{prefix}")
            for k in feat_dict:
                feat = feat_dict[k]
                builder.add_item({'item_name': k, 'feat': feat})
            builder.finalize()

        save_feat(train_set, 'train', hparams['binary_data_dir'])
        save_feat(valid_set, 'valid', hparams['binary_data_dir'])
        save_feat(test_set, 'test', hparams['binary_data_dir'])

