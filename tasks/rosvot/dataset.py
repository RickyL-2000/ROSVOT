import os
import math
import gc

import librosa.feature
import numpy as np
import torch
from tqdm import tqdm

from utils.commons.dataset_utils import BaseDataset, collate_1d_or_2d, pad_or_cut_xd
from utils.commons.indexed_datasets import IndexedDataset
from utils.audio import librosa_wav2spec
from utils.audio.pitch_utils import norm_interp_f0, denorm_f0, f0_to_coarse
from utils.commons.signal import get_filter_1d, get_gaussian_kernel_1d, get_hann_kernel_1d, \
    get_triangle_kernel_1d, add_gaussian_noise
from utils.audio.mel import MelNet

def get_soft_label_filter(soft_label_func, win_size, hparams):
    # win_size: ms
    win_size = round(int(win_size) * hparams['audio_sample_rate'] / 1000 / hparams['hop_size'])
    win_size = win_size if win_size % 2 == 1 else win_size + 1  # ensure odd number
    if soft_label_func == 'gaussian':
        sigma = win_size / 3 / 2  # 3sigma range
        kernel = get_gaussian_kernel_1d(win_size, sigma)
        kernel = kernel / kernel.max()  # make sure the middle is 1
    elif soft_label_func == 'hann':
        kernel = get_hann_kernel_1d(win_size, periodic=False)
    elif soft_label_func == 'triangle':
        kernel = get_triangle_kernel_1d(win_size)
    soft_filter = get_filter_1d(kernel, win_size, channels=1)
    return soft_filter

def get_mel_len(wav_len, hop_size):
    return (wav_len + hop_size - 1) // hop_size

class MidiDataset(BaseDataset):
    def __init__(self, prefix, shuffle=False, items=None, data_dir=None):
        super(MidiDataset, self).__init__(shuffle)
        from utils.commons.hparams import hparams
        self.data_dir = hparams['binary_data_dir'] if data_dir is None else data_dir
        self.prefix = prefix
        self.hparams = hparams
        self.indexed_ds = None
        if items is not None:
            self.indexed_ds = items
            self.sizes = [1] * len(items)
            self.avail_idxs = list(range(len(self.sizes)))
        else:
            self.sizes = np.load(f'{self.data_dir}/{self.prefix}_lengths.npy')
            if prefix == 'test' and len(hparams['test_ids']) > 0:
                self.avail_idxs = hparams['test_ids']
            else:
                self.avail_idxs = list(range(len(self.sizes)))
            self.sizes = [self.sizes[i] for i in self.avail_idxs]

        if items is None and self.avail_idxs is not None:
            ds_names_selected = None
            if prefix in ['train', 'valid'] and self.hparams.get('ds_names_in_training', '') != '':
                ds_names_selected = self.hparams.get('ds_names_in_training', '').split(';') + ['']
                print(f'| Iterating training sets to find samples belong to datasets {ds_names_selected[:-1]}')
            elif prefix == 'test' and self.hparams.get('ds_names_in_testing', '') != '':
                ds_names_selected = self.hparams.get('ds_names_in_testing', '').split(';') + ['']
                print(f'| Iterating testing sets to find samples belong to datasets {ds_names_selected[:-1]}')
            if ds_names_selected is not None:
                avail_idxs = []
                # somehow, can't use self.indexed_ds beforehand (need to create a temp), otherwise '_pickle.UnpicklingError'
                temp_ds = IndexedDataset(f'{self.data_dir}/{self.prefix}')
                for idx in tqdm(range(len(self)), total=len(self)):
                    item = temp_ds[self.avail_idxs[idx]]
                    if item.get('ds_name', '') in ds_names_selected:
                        avail_idxs.append(self.avail_idxs[idx])
                print(f'| Chose [{len(avail_idxs)}] samples belonging to the desired datasets from '
                      f'[{len(self.avail_idxs)}] original samples. ({len(avail_idxs) / len(self.avail_idxs) * 100:.2f}%)')
                self.avail_idxs = avail_idxs
        if items is None and prefix == 'train' and self.hparams.get('dataset_downsample_rate', 1.0) < 1.0 \
                and self.avail_idxs is not None:
            ratio = self.hparams.get('dataset_downsample_rate', 1.0)
            orig_len = len(self.avail_idxs)
            tgt_len = round(orig_len * ratio)
            self.avail_idxs = np.random.choice(self.avail_idxs, size=tgt_len, replace=False).tolist()
            print(f'| Downsamping training set with ratio [{ratio * 100:.2f}%], '
                  f'[{tgt_len}] samples of [{orig_len}] samples are selected.')
        if items is None:
            self.sizes = [self.sizes[i] for i in self.avail_idxs]

        self.soft_filter = {}
        self.noise_ds = None
        noise_snr = self.hparams.get('noise_snr', '6-20')
        if '-' in noise_snr:
            l, r = noise_snr.split('-')
            self.noise_snr = (float(l), float(r))
        else:
            self.noise_snr = float(noise_snr)
        self.mel_net = MelNet(self.hparams)

    def add_noise(self, clean_wav):
        if self.noise_ds is None:   # each instance in multiprocessing must create unique ds object
            self.noise_ds = IndexedDataset(f"{self.hparams['noise_data_dir']}/{self.prefix}")
        noise_idx = np.random.randint(len(self.noise_ds))
        noise_item = self.noise_ds[noise_idx]
        noise_wav = noise_item['feat']

        if type(self.noise_snr) == tuple:
            snr = np.random.rand() * (self.noise_snr[1] - self.noise_snr[0]) + self.noise_snr[0]
        else:
            snr = self.noise_snr
        clean_rms = np.sqrt(np.mean(np.square(clean_wav), axis=-1))
        if len(clean_wav) > len(noise_wav):
            ratio = int(np.ceil(len(clean_wav)/len(noise_wav)))
            noise_wav = np.concatenate([noise_wav for _ in range(ratio)])
        if len(clean_wav) < len(noise_wav):
            start = 0
            noise_wav = noise_wav[start: start + len(clean_wav)]
        noise_rms = np.sqrt(np.mean(np.square(noise_wav), axis=-1)) + 1e-5
        adjusted_noise_rms = clean_rms / (10 ** (snr / 20) + 1e-5)
        adjusted_noise_wav = noise_wav * (adjusted_noise_rms / noise_rms)
        mixed = clean_wav + adjusted_noise_wav
        # Avoid clipping noise
        max_int16 = np.iinfo(np.int16).max
        min_int16 = np.iinfo(np.int16).min
        if mixed.max(axis=0) > max_int16 or mixed.min(axis=0) < min_int16:
            if mixed.max(axis=0) >= abs(mixed.min(axis=0)):
                reduction_rate = max_int16 / mixed.max(axis=0)
            else:
                reduction_rate = min_int16 / mixed.min(axis=0)
            mixed = mixed * reduction_rate
        return mixed

    def _get_item(self, index):
        if hasattr(self, 'avail_idxs') and self.avail_idxs is not None:
            index = self.avail_idxs[index]
        if self.indexed_ds is None:
            self.indexed_ds = IndexedDataset(f'{self.data_dir}/{self.prefix}')
        return self.indexed_ds[index]

    def __getitem__(self, index):
        hparams = self.hparams
        item = self._get_item(index)

        wav = item['wav']
        noise_added = np.random.rand() < hparams.get('noise_prob', 0.8)
        if self.prefix == 'test' and not hparams.get('noise_in_test', False):
            noise_added = False
        if noise_added:
            wav = self.add_noise(wav)

        mel = self.mel_net(wav).squeeze(0).numpy()
        assert len(mel) == self.sizes[index], (len(mel), self.sizes[index])
        max_frames = hparams['max_frames']
        mel2word_len = sum((item["mel2word"] > 0).astype(int))
        T = min(item['len'], mel2word_len, len(item['f0']))
        real_len = T
        T = math.ceil(min(T, max_frames) / hparams['frames_multiple']) * hparams['frames_multiple']
        spec = torch.Tensor(mel)[:max_frames]
        spec = pad_or_cut_xd(spec, T, dim=0)
        if 5 < hparams.get('use_mel_bins', hparams['audio_num_mel_bins']) < hparams['audio_num_mel_bins']:
            spec = spec[:, :hparams.get('use_mel_bins', 80)]
        sample = {
            "id": index,
            "item_name": item['item_name'],
            "mel": spec,
            "mel_nonpadding": spec.abs().sum(-1) > 0,
        }
        if hparams.get('mel_add_noise', 'none') not in ['none', None] and not noise_added \
                and self.prefix in ['train', 'valid']:
            noise_type, std = hparams.get('mel_add_noise').split(':')
            if noise_type == 'gaussian':
                noisy_mel = add_gaussian_noise(sample['mel'], mean=0.0, std=float(std) * np.random.rand())
                sample['mel'] = torch.clamp(noisy_mel, hparams['mel_vmin'], hparams['mel_vmax'])
        sample["mel2word"] = mel2word = pad_or_cut_xd(torch.LongTensor(item.get("mel2word")), T, 0)
        sample['mel_nonpadding'] = pad_or_cut_xd(sample['mel_nonpadding'].float(), T, 0)
        if hparams['use_pitch_embed']:
            assert 'f0' in item
            # pitch = torch.LongTensor(item.get(hparams.get('pitch_key', 'pitch')))[:T]
            f0, uv = norm_interp_f0(item["f0"][:T])
            if hparams.get('f0_add_noise', 'none') not in ['none', None] and noise_added \
                    and self.prefix in ['train', 'valid']:
                noise_type, std = hparams.get('f0_add_noise').split(':')
                f0 = torch.FloatTensor(f0)
                if noise_type == 'gaussian':
                    f0[uv == 0] = add_gaussian_noise(f0[uv == 0], mean=0.0, std=float(std) * np.random.rand())
            uv = pad_or_cut_xd(torch.FloatTensor(uv), T, 0)
            f0 = pad_or_cut_xd(torch.FloatTensor(f0), T, 0)
            pitch_coarse = f0_to_coarse(denorm_f0(f0, uv))
        else:
            f0, uv, pitch, pitch_coarse = None, None, None, None
        sample["f0"], sample["uv"], sample["pitch_coarse"] = f0, uv, pitch_coarse
        sample["note"] = torch.LongTensor(item['pitches'][:hparams['max_input_tokens']])
        sample["word_dur"] = torch.FloatTensor(item['word_durs'][:hparams['max_input_tokens']])
        sample["note_dur"] = torch.FloatTensor(item['note_durs'][:hparams['max_input_tokens']])

        # make boundary labels for word
        word_bd = torch.zeros_like(mel2word)
        for idx in range(1, real_len):
            if mel2word[idx] != mel2word[idx-1]:
                word_bd[idx] = 1.
        sample["word_bd"] = word_bd.long()
        # assert sample["word_dur"].shape[0] == torch.sum(sample["word_bd"]) + 1, f"{sample['word_dur'].shape[0]} {torch.sum(sample['word_bd']) + 1}"
        if sample["word_dur"].shape[0] != torch.sum(sample["word_bd"]) + 1:
            def bd_to_idxs(bd):
                # bd [T]
                idxs = []
                for idx in range(len(bd)):
                    if bd[idx] == 1:
                        idxs.append(idx)
                return idxs
            print(sample['word_dur'].shape[0], torch.sum(sample['word_bd']) + 1)
            print(sample['item_name'])
            print(sample['word_dur'])
            print(bd_to_idxs(sample["word_bd"]))

        # make boundary labels for note
        note_bd = torch.zeros_like(mel2word)
        note_dur_ = sample["note_dur"].cumsum(0)[:-1]
        note_bd_idx = torch.round(note_dur_ * hparams['audio_sample_rate'] / hparams["hop_size"]).long()
        note_bd_max_idx = real_len - 1
        note_bd[note_bd_idx[note_bd_idx < note_bd_max_idx]] = 1
        sample["note_bd"] = note_bd.long()
        # deal with truncated note boundaries and the corresponding notes and note durs
        if note_bd.sum() + 1 < len(sample['note']):
            tgt_size = note_bd.sum().item() + 1
            sample['note'] = sample['note'][:tgt_size]
            sample['note_dur'] = sample['note_dur'][:tgt_size]
        if hparams.get('use_soft_note_bd', False) and (hparams.get('soft_note_bd_func', None) not in ['none', None]):
            if 'note_bd' not in self.soft_filter:
                soft_label_func, win_size = hparams.get('soft_note_bd_func', None).split(':')
                self.soft_filter['note_bd'] = get_soft_label_filter(soft_label_func, int(win_size), hparams)
            note_bd_soft = note_bd.clone().detach().float()
            note_bd_soft[note_bd.eq(1)] = note_bd_soft[note_bd.eq(1)] - 1e-7  # avoid nan
            note_bd_soft = note_bd_soft.unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                note_bd_soft = self.soft_filter['note_bd'](note_bd_soft).squeeze().detach()
            sample['note_bd_soft'] = note_bd_soft
            if hparams.get('note_bd_add_noise', 'none') not in ['none', None]:
                noise_type, std = hparams.get('note_bd_add_noise').split(':')
                if noise_type == 'gaussian':
                    noisy_note_bd_soft = add_gaussian_noise(note_bd_soft, mean=0.0, std=float(std) * np.random.rand())
                    sample['note_bd_soft'] = torch.clamp(noisy_note_bd_soft, 0.0, 1 - 1e-7)

        # delete big redundancy
        if not hparams.get('use_mel', True) and 'mel' in sample:
            del sample['mel']
            # if 'mel_nonpadding' in sample:
            #     sample['mel_nonpadding'] = None
        if not hparams.get('use_wav', False) and 'wav' in sample:
            del sample['wav']

        return sample

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        hparams = self.hparams
        id = torch.LongTensor([s['id'] for s in samples])
        item_names = [s['item_name'] for s in samples]
        # text = [s['text'] for s in samples]
        mels = collate_1d_or_2d([s['mel'] for s in samples], 0.0) if 'mel' in samples[0] else None
        mel_lengths = torch.LongTensor([s['mel'].shape[0] for s in samples]) if 'mel' in samples[0] else 0
        mel_nonpadding = collate_1d_or_2d([s['mel_nonpadding'] for s in samples], 0.0) if 'mel_nonpadding' in samples[0] else None

        batch = {
            'id': id,
            'item_name': item_names,
            'nsamples': len(samples),
            'mels': mels,
            'mel_lengths': mel_lengths,
            'mel_nonpadding': mel_nonpadding
        }

        if hparams['use_pitch_embed']:
            f0 = collate_1d_or_2d([s['f0'] for s in samples], 0.0)
            uv = collate_1d_or_2d([s['uv'] for s in samples])
            pitch_coarse = collate_1d_or_2d([s['pitch_coarse'] for s in samples])
        else:
            f0, uv, pitch, pitch_coarse = None, None, None, None
        batch['f0'], batch['uv'], batch['pitch_coarse'] = f0, uv, pitch_coarse
        batch["wav"] = collate_1d_or_2d([s['wav'] for s in samples], 0.0) if 'wav' in samples[0] else None

        batch['mel2word'] = collate_1d_or_2d([s['mel2word'] for s in samples], 0)
        batch["notes"] = collate_1d_or_2d([s['note'] for s in samples], 0.0)
        batch["note_durs"] = collate_1d_or_2d([s['note_dur'] for s in samples], 0.0)
        batch["word_durs"] = collate_1d_or_2d([s['word_dur'] for s in samples], 0.0)
        batch["word_bd"] = collate_1d_or_2d([s['word_bd'] for s in samples], 0.0)
        batch["note_bd"] = collate_1d_or_2d([s['note_bd'] for s in samples], 0.0)
        batch["word_bd_soft"] = collate_1d_or_2d([s['word_bd_soft'] for s in samples], 0.0) if 'word_bd_soft' in samples[0] else None
        batch["note_bd_soft"] = collate_1d_or_2d([s['note_bd_soft'] for s in samples], 0.0) if 'note_bd_soft' in samples[0] else None

        return batch

    @property
    def num_workers(self):
        # if self.prefix == 'train':
        #     return int(os.getenv('NUM_WORKERS', self.hparams['ds_workers']))
        # return 1
        return int(os.getenv('NUM_WORKERS', self.hparams['ds_workers']))


class WordbdDataset(MidiDataset):
    def __getitem__(self, index):
        sample = super(WordbdDataset, self).__getitem__(index)
        hparams = self.hparams

        # make soft word label
        if hparams.get('use_soft_word_bd', False) and (hparams.get('soft_word_bd_func', None) not in ['none', None]):
            if 'word_bd' not in self.soft_filter:
                soft_label_func, win_size = hparams.get('soft_word_bd_func', None).split(':')
                self.soft_filter['word_bd'] = get_soft_label_filter(soft_label_func, int(win_size), hparams)
            word_bd = sample['word_bd']
            word_bd_soft = word_bd.clone().detach().float()
            word_bd_soft[word_bd.eq(1)] = word_bd_soft[word_bd.eq(1)] - 1e-7  # avoid nan
            word_bd_soft = word_bd_soft.unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                word_bd_soft = self.soft_filter['word_bd'](word_bd_soft).squeeze().detach()
            sample['word_bd_soft'] = word_bd_soft
            if hparams.get('word_bd_add_noise', 'none') not in ['none', None]:
                noise_type, std = hparams.get('word_bd_add_noise').split(':')
                if noise_type == 'gaussian':
                    noisy_word_bd_soft = add_gaussian_noise(word_bd_soft, mean=0.0, std=float(std) * np.random.rand())
                    sample['word_bd_soft'] = torch.clamp(noisy_word_bd_soft, 0.0, 1 - 1e-7)

        return sample
