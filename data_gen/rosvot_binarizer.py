import random, os
import subprocess
from copy import deepcopy
import logging
import json
from functools import partial
from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm
import pyworld as pw
import torch

from data_gen.base_binarizer import BaseBinarizer, BinarizationError
from utils.commons.hparams import hparams
from utils.commons.indexed_datasets import IndexedDatasetBuilder
from utils.commons.multiprocess_utils import chunked_multiprocess_run
from utils.audio.align import mel2token_to_dur
from utils.audio.pitch_utils import f0_to_coarse, resample_align_curve, hz_to_midi
from utils.commons.dataset_utils import pad_or_cut_xd
from utils.audio.mel import MelNet
from modules.pe.rmvpe import RMVPE
import modules.pe.rmvpe.extractor as f0_extractor

rmvpe = None
f0_dict = None

class RosvotBinarizer(BaseBinarizer):
    def __init__(self, processed_data_dir=None):
        super(RosvotBinarizer, self).__init__(processed_data_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        from utils.commons.hparams import hparams
        self.mel_net = MelNet(hparams)
        self.hparams = hparams

    def split_train_test_set(self, item_names):
        item_names = deepcopy(item_names)
        test_item_names = [x for x in item_names if any([ts in x for ts in hparams['test_prefixes']])]
        train_item_names = [x for x in item_names if x not in set(test_item_names)]
        logging.info("train {}".format(len(train_item_names)))
        logging.info("test {}".format(len(test_item_names)))
        return train_item_names, test_item_names

    def load_meta_data(self):
        metafile_path = hparams.get('metafile_path', f"{self.processed_data_dir}/metadata.json")
        if ',' in metafile_path:
            metafile_paths = metafile_path.split(',')
            ds_names = hparams.get('ds_names', ','.join([str(i) for i in range(len(metafile_paths))])).split(',')
        else:
            metafile_paths = [metafile_path]
            ds_names = [hparams.get('ds_names', '0')]
        for idx, metafile_path in enumerate(metafile_paths):
            items_list = json.load(open(metafile_path))
            for r in tqdm(items_list, desc=f'| Loading meta data for dataset {ds_names[idx]}.'):
                item_name = r['item_name']
                if item_name in self.items:
                    print(f'warning: item name {item_name} duplicated')
                self.items[item_name] = r
                self.item_names.append(item_name)
                self.items[item_name]['ds_name'] = ds_names[idx]
        if self.binarization_args['shuffle']:
            random.seed(hparams.get('seed', 42))
            random.shuffle(self.item_names)
        self._train_item_names, self._test_item_names = self.split_train_test_set(self.item_names)

    @property
    def train_item_names(self):
        return self._train_item_names

    @property
    def valid_item_names(self):
        return self._test_item_names

    @property
    def test_item_names(self):
        # valid and test are the same in training, for inference check ./inference
        return self._test_item_names

    def process(self):
        self.load_meta_data()
        os.makedirs(hparams['binary_data_dir'], exist_ok=True)
        self.process_data('valid')
        self.process_data('test')
        self.process_data('train')

    @torch.no_grad()
    def process_data(self, prefix):
        data_dir = hparams['binary_data_dir']
        args = []
        builder = IndexedDatasetBuilder(f'{data_dir}/{prefix}')
        process_item = partial(self.process_item, hparams=hparams)
        lengths = []
        total_sec = 0
        meta_data = list(self.meta_data(prefix))
        args = [item for item in meta_data]

        # extract f0
        if hparams.get('pe', 'pw') == 'rmvpe':
            wav_fns = [item['wav_fn'] for item in args]
            f0s = f0_extractor.extract(wav_fns, ckpt=hparams.get('pe_ckpt', None), sr=hparams['audio_sample_rate'],
                                       hop_size=hparams['hop_size'], fmax=hparams['f0_max'], fmin=hparams['f0_min'],
                                       ds_workers=8)
            args = [{**item, **{'f0': f0s[idx]}} for idx, item in enumerate(args)]

        for item_id, (_, item) in enumerate(    # to suit for 'spawn', num_workers should be smaller for speed balance
                zip(tqdm(meta_data, desc='| Processing data', total=len(args)),
                    chunked_multiprocess_run(process_item, args, num_workers=16 if prefix == 'train' else 2))):
        # NOTE: if above 3 lines don't work, comment them and uncomment the following 2 lines for single processing
        # for arg in tqdm(args, desc='| Processing data', total=len(args)):
        #     item = process_item(arg)
            if item is None:
                continue
            if item['f0'] is None:
                print(f"warning: item {item['item_name']} has None f0.")
            if not self.binarization_args['with_wav'] and 'wav' in item:
                del item['wav']
            if not self.binarization_args['with_mel'] and 'mel' in item:
                del item['mel']

            builder.add_item(item)
            lengths.append(item['len'])
            total_sec += item['sec']
        builder.finalize()
        np.save(f'{data_dir}/{prefix}_lengths.npy', lengths)
        print(f"| {prefix} total duration: {total_sec:.3f}s")

    def process_audio(self, wav_fn, res, audio_sample_rate):
        sample_rate = audio_sample_rate
        wav, _ = librosa.load(wav_fn, sr=sample_rate, mono=True)
        mel = self.mel_net(wav).numpy().squeeze(0)  # [T, 80]
        res.update({'mel': mel, 'wav': wav, 'sec': len(wav) / audio_sample_rate, 'len': mel.shape[0]})
        return wav, mel

    @torch.no_grad()
    def process_item(self, item, hparams):
        item_name = item['item_name']
        wav_fn = item['wav_fn']
        wav, mel = self.process_audio(wav_fn, item, hparams['audio_sample_rate'])
        length = mel.shape[0]
        try:
            if hparams.get('f0_filepath', '') == '':
                f0 = None
                pe = hparams.get('pe', 'pw')
                if pe == 'rmvpe':
                    pass
                elif pe == 'pw':
                    f0, _ = pw.harvest(wav.astype(np.double), hparams['audio_sample_rate'],
                                       frame_period=hparams['hop_size'] * 1000 / hparams['audio_sample_rate'])
                    delta_l = length - len(f0)
                    if delta_l < 0:
                        f0 = f0[:length]
                    elif delta_l > 0:
                        f0 = np.concatenate((f0, np.full(delta_l, fill_value=f0[-1])), axis=0)
            else:
                global f0_dict
                if f0_dict is None:
                    f0_dict = np.load(hparams.get('f0_filepath', ''), allow_pickle=True).item()
                f0 = f0_dict[item_name]
                if abs(f0.shape[0] - length) < 5:
                    f0 = pad_or_cut_xd(torch.from_numpy(f0), length).numpy()
                assert len(f0) == length

            if f0 is not None:
                item['f0'] = f0

            mel2word, dur_word = align_word(item['word_durs'], mel.shape[0], hparams['hop_size'], hparams['audio_sample_rate'])
            if mel2word[0] == 0:    # better start from 1, consistent with mel2ph
                mel2word = [i + 1 for i in mel2word]
            item['mel2word'], item['dur_word'] = mel2word, dur_word

        except BinarizationError as e:
            print(f"| Skip item ({e}). item_name: {item_name}, wav_fn: {wav_fn}")
            return None
        return item

def align_word(word_durs, mel_len, hop_size, audio_sample_rate):
    mel2word = np.zeros([mel_len], int)
    start_time = 0
    for i_word in range(len(word_durs)):
        start_frame = int(start_time * audio_sample_rate / hop_size + 0.5)
        end_frame = int((start_time + word_durs[i_word]) * audio_sample_rate / hop_size + 0.5)
        mel2word[start_frame:end_frame] = i_word + 1
        start_time = start_time + word_durs[i_word]

    dur_word = mel2token_to_dur(mel2word)

    return mel2word, dur_word.tolist()
