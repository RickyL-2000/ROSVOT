import os
import argparse
from pathlib import Path
import json
import math
from collections import defaultdict
import traceback
import sys

import librosa
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.multiprocessing as mp
from torch.distributed import init_process_group
import torch.distributed as dist
import matplotlib.pyplot as plt

from utils.os_utils import safe_path
from utils.commons.hparams import set_hparams
from utils.commons.multiprocess_utils import MultiprocessManager
from utils.commons.dataset_utils import batch_by_size, pad_or_cut_xd, collate_1d_or_2d, build_dataloader
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.tensor_utils import move_to_cuda
from utils.audio import get_wav_num_frames
from utils.audio.mel import MelNet
from utils.audio.pitch_utils import norm_interp_f0, denorm_f0, f0_to_coarse, boundary2Interval, save_midi, midi_to_hz

from modules.pe.rmvpe import RMVPE
from data_gen.rosvot_binarizer import align_word
from tasks.rosvot.dataset import get_mel_len
from tasks.rosvot.rosvot_utils import bd_to_durs, regulate_real_note_itv, regulate_ill_slur
from modules.rosvot.rosvot import MidiExtractor, WordbdExtractor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--save_dir",
        type=str,
        help='Directory of outputs. '
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default='checkpoints/rosvot/model.pt',
        help='Path of rosvot ckpt. '
    )
    parser.add_argument(
        "--config",
        type=str,
        default='',
        help='Path of config file. If not provided, will be inferred under the same directory of ckpt. '
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default='',
        help='Path of the metadata of the desired input data. '
             'The metadata should be a .json file containing a list of dicts, where each dicts should contain'
             'at least two attributes: "item_name", and "wav_fn". '
             'Another attribute "word_durs" could also be contained.'
    )
    parser.add_argument(
        "-p",
        "--wav_fn",
        type=str,
        default='',
        help='Path of the input wav file. '
    )
    parser.add_argument(
        "--thr",
        type=float,
        default=0.85,
        help='Threshold to determine note boundaries. '
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action='store_true',
        help="Whether or not to print detailed information. "
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=30000,
        help='Max frames for one sample. '
    )
    parser.add_argument(
        "--bsz",
        type=int,
        default=128,
        help='Batch size (max sentences) for each node. '
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=100000,
        help='Max tokens for each node. '
    )
    parser.add_argument(
        "--ds_workers",
        type=int,
        default=1,
        help='Number of workers to generate samples. Set to 0 for single inference. '
    )
    parser.add_argument(
        "--apply_rwbd",
        action='store_true',
        help="Force to apply word boundary predictor, even if word_durs is available. "
    )
    parser.add_argument(
        "--wbd_ckpt",
        type=str,
        default='checkpoints/rwbd/model.pt',
        help='Path of word boundary predictor ckpt. '
    )
    parser.add_argument(
        "--wbd_config",
        type=str,
        default='',
        help='Path of config file of word boundary predictor. '
             'If not provided, will be inferred under the same directory of ckpt.'
    )
    parser.add_argument(
        "--wbd_thr",
        type=float,
        default=0.5,
        help='Threshold to determine word boundaries. '
    )
    parser.add_argument(
        "--save_plot",
        action='store_true',
        help='Save the plots of MIDI or not. '
    )
    parser.add_argument(
        "--check_slur",
        action='store_true',
        help='Check appearances of slurs and print logs.'
    )
    parser.add_argument(
        "--no_save_midi",
        action='store_true',
        help="Don't save MIDI files. "
    )
    parser.add_argument(
        "--no_save_every_npy",
        action='store_true',
        help="Don't save npy files for each sample. "
    )
    parser.add_argument(
        "--no_save_final_npy",
        action='store_true',
        help="Don't save the last final npy file. "
    )
    parser.add_argument(
        "--sync_saving",
        action='store_true',
        help="Synchronized results saving. "
    )
    args = parser.parse_args()

    return args

class RosvotInfer:
    def __init__(self, num_gpus=-1):
        self.args = parse_args()
        self.work_dir = self.args.save_dir
        if num_gpus == -1:
            all_gpu_ids = [int(x) for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x != '']
            self.num_gpus = len(all_gpu_ids)
        else:
            self.num_gpus = num_gpus
        self.hparams = {}
        self.apply_rwbd = True

    def build_model(self, device=None, verbose=True):
        model = MidiExtractor(self.hparams)
        load_ckpt(model, self.args.ckpt, verbose=verbose)
        model.eval()
        if device is not None:
            model.to(device)
        wbd_predictor = None
        if self.apply_rwbd:
            wbd_ckpt_path = self.args.wbd_ckpt
            wbd_config_path = Path(wbd_ckpt_path).with_name('config.yaml') \
                if self.args.wbd_config == '' else self.args.wbd_config
            wbd_hparams = set_hparams(
                config=wbd_config_path,
                print_hparams=False,
                hparams_str=f"max_frames={self.args.max_frames},"
                            f"word_bd_threshold={self.args.wbd_thr}"
            )
            self.hparams['wbd_use_mel_bins'] = wbd_hparams['use_mel_bins']
            self.hparams.update({
                'wbd_use_mel_bins': wbd_hparams['use_mel_bins'],
                'min_word_dur': wbd_hparams['min_word_dur'],
            })
            wbd_predictor = WordbdExtractor(wbd_hparams)
            load_ckpt(wbd_predictor, wbd_ckpt_path, verbose=verbose)
            wbd_predictor.eval()
            if device is not None:
                wbd_predictor.to(device)
        return model, wbd_predictor

    def run(self):
        ckpt_path = self.args.ckpt
        config_path = Path(ckpt_path).with_name('config.yaml') if self.args.config == '' else self.args.config
        self.hparams = set_hparams(
            config=config_path,
            print_hparams=self.args.verbose,
            hparams_str=f"max_frames={self.args.max_frames},"
                        f"note_bd_threshold={self.args.thr}"
        )

        if self.args.metadata:
            items = json.load(open(self.args.metadata))
            if 'word_durs' in items[0]:
                self.apply_rwbd = False
        else:
            assert self.args.wav_fn != '', "--metadata and --wav_fn cannot both be empty"
            items = [{'item_name': 'output', 'wav_fn': self.args.wav_fn}]
            self.args.ds_workers = 0
        if self.args.apply_rwbd:
            self.apply_rwbd = True

        results = []
        if self.num_gpus > 1:
            result_queue = mp.Queue()
            for rank in range(self.num_gpus):
                mp.Process(target=self.run_worker, args=(rank, items, self.args.ds_workers, not self.args.sync_saving, result_queue,)).start()
            for _ in range(self.num_gpus):
                results_ = result_queue.get()
                results.extend(results_)
        else:
            results = self.run_worker(0, items, self.args.ds_workers, False, None)

        results = self.after_infer(results)

        return results

    @torch.no_grad()
    def run_worker(self, rank, items, ds_workers=0, async_save_result=True, q=None):
        if self.num_gpus > 1:
            init_process_group(backend="nccl", init_method="tcp://localhost:54189",
                               world_size=self.num_gpus, rank=rank)

        # build models
        device = torch.device(f"cuda:{int(rank)}")
        # build mel model
        mel_net = MelNet(self.hparams)
        mel_net.to(device)
        # build f0 model
        if self.hparams['use_pitch_embed']:
            pe = RMVPE(self.hparams['pe_ckpt'], device=device)
        # build main model
        model, wbd_predictor = self.build_model(device, verbose=rank == 0)

        # build dataset
        dataset = RosvotInferDataset(items, self.hparams, ds_workers)
        loader = build_dataloader(dataset, shuffle=False, max_tokens=self.args.max_tokens, max_sentences=self.args.bsz,
                                  use_ddp=self.num_gpus > 1)
        loader = tqdm(loader, desc=f"| Generating in [n_ranks={self.num_gpus}; "
                                   f"max_tokens={self.args.max_tokens}; "
                                   f"max_sentences={self.args.bsz}]") if rank == 0 else loader

        # results queue
        saving_result_pool = MultiprocessManager(int(ds_workers))
        results = []

        # run main inference
        with torch.no_grad():
            for batch in loader:
                if batch is None or len(batch) == 0:
                    continue
                batch = move_to_cuda(batch, int(rank))
                bsz = batch['nsamples']

                # get f0
                if self.hparams['use_pitch_embed']:
                    f0s, uvs = pe.get_pitch_batch(
                        batch['wav'], sample_rate=self.hparams['audio_sample_rate'], hop_size=self.hparams['hop_size'],
                        lengths=batch['real_lens'], fmax=self.hparams['f0_max'], fmin=self.hparams['f0_min']
                    )
                    f0_batch, uv_batch, pitch_batch = [], [], []
                    for i, (f0, uv) in enumerate(zip(f0s, uvs)):
                        T = batch['lens'][i]
                        f0, uv = norm_interp_f0(f0[:T])
                        f0 = pad_or_cut_xd(torch.FloatTensor(f0), T, 0)
                        f0_batch.append(f0)
                        uv = pad_or_cut_xd(torch.FloatTensor(uv), T, 0)
                        uv_batch.append(uv)
                        pitch_batch.append(f0_to_coarse(denorm_f0(f0, uv)))
                    batch["f0"] = f0 = collate_1d_or_2d(f0_batch).to(device)
                    batch["uv"] = uv = collate_1d_or_2d(uv_batch).long().to(device)
                    batch["pitch_coarse"] = pitch_coarse = collate_1d_or_2d(pitch_batch).to(device)
                else:
                    batch["f0"] = f0 = batch["uv"] = uv = batch["pitch_coarse"] = pitch_coarse = None

                # get mel
                mel = mel_net(batch['wav'])     # [B, T, C]
                mel = pad_or_cut_xd(mel, max(batch['lens']), 1)
                mel = (mel.transpose(1, 2) * batch['mel_nonpadding'].unsqueeze(1)).transpose(1, 2)
                batch['mel_nonpadding'] = mel_nonpadding = mel.abs().sum(-1) > 0
                batch['mel'] = mel

                # get wbd
                if self.apply_rwbd:
                    mel_input = mel[:, :, :self.hparams.get('wbd_use_mel_bins', 80)]
                    wbd_outputs = wbd_predictor(mel=mel_input, pitch=pitch_coarse, uv=uv, non_padding=mel_nonpadding, train=False)
                    word_bd = wbd_outputs['word_bd_pred']  # [B, T]
                else:
                    word_bd = batch['word_bd']

                # get midi
                mel_input = mel[:, :, :self.hparams.get('use_mel_bins', 80)]
                outputs = model(mel=mel_input, word_bd=word_bd, pitch=pitch_coarse, uv=uv, non_padding=mel_nonpadding, train=False)
                outputs['word_bd_pred'] = word_bd if self.apply_rwbd else None

                # postprocess
                rets = []
                real_lens = batch['real_lens']
                note_lengths = outputs['note_lengths'].cpu().numpy()
                for idx in range(bsz):
                    note_bd_pred = outputs['note_bd_pred'][idx].cpu().numpy()[:real_lens[idx]]
                    note_pred = outputs['note_pred'][idx].cpu().numpy()[:note_lengths[idx]]
                    f0 = denorm_f0(batch['f0'], batch['uv'])[idx].cpu().numpy()[:real_lens[idx]] if self.hparams['use_pitch_embed'] else None
                    item_name = batch['item_name'][idx]
                    note_bd_logits = torch.sigmoid(outputs['note_bd_logits'])[idx].data.cpu()[:real_lens[idx]]

                    # make word durs
                    if self.apply_rwbd:
                        word_bd = outputs['word_bd_pred'][idx].cpu().numpy()[:real_lens[idx]]
                        word_durs = np.array(bd_to_durs(word_bd)) * self.hparams['hop_size'] / self.hparams['audio_sample_rate']
                    else:
                        word_bd = batch['word_bd'][idx].cpu().numpy()[:real_lens[idx]]
                        word_durs = batch['word_durs'][idx][:batch['word_lens'][idx]].cpu().numpy()

                    ret = {
                        'item_name': item_name,
                        'note_bd_pred': note_bd_pred,
                        'note_pred': note_pred,
                        'f0': f0,
                        'word_bd': word_bd,
                        'word_durs': word_durs,
                        'note_bd_logits': note_bd_logits
                    }
                    rets.append(ret)

                # save results
                for i, output in enumerate(rets):
                    if async_save_result:
                        saving_result_pool.add_job(self.save_result, args=[output,])
                    else:
                        res = self.save_result(output)
                        results.append(res)

        if async_save_result:
            for res_id, res in saving_result_pool.get_results():
                results.append(res)

        if q is not None:
            q.put(results)
        else:
            return results

    def save_result(self, output):
        item_name = output['item_name']
        note_bd_pred = output['note_bd_pred']
        note_pred = output['note_pred']
        f0 = output['f0']
        word_bd = output['word_bd']
        word_durs = output['word_durs'] if 'word_durs' in output else None
        if note_pred.shape == (0,):
            if self.args.verbose:
                print(f"skip {item_name}: no notes detected")
            return
        fn = item_name
        note_itv_pred = boundary2Interval(note_bd_pred)
        if self.hparams.get('infer_regulate_real_note_itv', True) and not self.apply_rwbd:
            try:
                note_itv_pred_secs, note2words = regulate_real_note_itv(note_itv_pred, note_bd_pred, word_bd, word_durs, self.hparams['hop_size'], self.hparams['audio_sample_rate'])
                note_pred, note_itv_pred_secs, note2words = regulate_ill_slur(note_pred, note_itv_pred_secs, note2words)
                if not self.args.no_save_midi:
                    save_midi(note_pred, note_itv_pred_secs, safe_path(f'{self.work_dir}/midi/{fn}.mid'))
                check_slur_cnt(note2words, item_name, verbose=self.args.verbose)
            except Exception as err:
                if self.args.verbose:
                    _, exc_value, exc_tb = sys.exc_info()
                    tb = traceback.extract_tb(exc_tb)[-1]
                    print(f'skip {item_name}, {err}: {exc_value} in {tb[0]}:{tb[1]} "{tb[2]}" in {tb[3]}')
                return
        else:
            note_itv_pred_secs = note_itv_pred * self.hparams['hop_size'] / self.hparams['audio_sample_rate']
            if not self.args.no_save_midi:
                save_midi(note_pred, note_itv_pred_secs, safe_path(f'{self.work_dir}/midi/{fn}.mid'))
            note2words = None
        note_durs = []
        for itv in note_itv_pred_secs:
            # note_durs.append(round(itv[1] - itv[0], ndigits=3))
            note_durs.append(itv[1] - itv[0])
        out = {'item_name': item_name,
               'pitches': note_pred.tolist(),
               'note_durs': note_durs,
               'note2words': note2words.tolist() if note2words is not None else None}
        if not self.args.no_save_every_npy:
            np.save(safe_path(f'{self.work_dir}/npy/[note]{fn}.npy'), out, allow_pickle=True)

        if self.args.save_plot:
            fig = plt.figure()
            if f0 is not None:
                plt.plot(f0, color='red', label='f0')
            midi_pred = np.zeros(note_bd_pred.shape[0])
            for i, itv in enumerate(
                    np.round(note_itv_pred_secs * self.hparams['audio_sample_rate'] / self.hparams['hop_size']).astype(int)):
                midi_pred[itv[0]: itv[1]] = note_pred[i]
            midi_pred = midi_to_hz(midi_pred)
            plt.plot(midi_pred, color='blue', label='pred midi')
            note_bd_logits = output['note_bd_logits']
            np.save(safe_path(f'{self.work_dir}/npy/[bd]{fn}.npy'), note_bd_logits)
            plt.plot(note_bd_logits * 100, color='green', label='note bd logits')
            plt.legend()
            plt.tight_layout()
            plt.savefig(safe_path(f'{self.work_dir}/plot/[MIDI]{fn}.png'), format='png')
            plt.close(fig)

        return out

    def after_infer(self, results):
        d = {}
        slur_cnt = defaultdict(int)
        n_skipped = 0
        for r in results:
            if r is None:
                if not self.args.verbose:
                    print('A sample is skipped for some reasons. Turn on the --verbose flag for detailed logs.')
                n_skipped += 1
                continue
            d[r['item_name']] = r
            item_name, note2words = r['item_name'], r['note2words']
            if note2words is not None and self.args.check_slur:
                _slur_cnt = check_slur_cnt(note2words)
                for k in _slur_cnt:
                    slur_cnt[k] += _slur_cnt[k]
        slur_cnt = dict(sorted(slur_cnt.items(), key=lambda x: x[0]))
        if (not self.apply_rwbd) and self.args.check_slur:
            if len(slur_cnt) > 0:
                print('| Statistics for slurs: there are averagely')
                for k in slur_cnt:
                    print(f"  [{slur_cnt[k] / len(results):.3f}] instances of [{k}]-slur in each sample. | [{slur_cnt[k]}] instances in total.")
            else:
                print('| No slurs detected.')
        print(f"| Totally {len(d)} items saved, {n_skipped} items skipped.")
        if not self.args.no_save_final_npy:
            np.save(safe_path(f'{self.work_dir}/notes.npy'), d, allow_pickle=True)
        return d

class RosvotInferDataset(Dataset):
    def __init__(self, items, hparams, num_workers):
        self.sizes = []
        self.hparams = hparams
        self.hop_size = hparams['hop_size']
        self.sr = hparams['audio_sample_rate']
        if type(items[0]['wav_fn']) == str:  # wav_paths
            for idx, item in enumerate(items):
                total_frames = get_wav_num_frames(item['wav_fn'], self.sr)
                self.sizes.append(get_mel_len(total_frames, self.hop_size))
        else:  # numpy arrays, mono wavs
            for idx, item in enumerate(items):
                self.sizes.append(get_mel_len(item['wav_fn'].shape[-1], self.hop_size))
        self.num_workers = num_workers
        self.items = items
        # self.mel_net = MelNet(self.hparams)

    def __getitem__(self, idx):
        hparams = self.hparams
        item = self.items[idx]
        item_name = item['item_name']
        if type(item['wav_fn']) == str:
            wav_fn = item['wav_fn']
            wav, _ = librosa.core.load(wav_fn, sr=self.sr)
        else:
            wav = item['wav_fn']
        # mel = self.mel_net(wav).squeeze(0).numpy()  # [T, C]  # compute mel in the forward step
        mel_len = get_mel_len(wav.shape[-1], self.hop_size)

        if 'word_durs' in item:
            # filter invalid word dur
            word_durs = []
            for i in range(len(item['word_durs'])):
                wd = item['word_durs'][i]
                if wd < hparams.get('min_word_dur', 20) / 1000:
                    print(f'| warning: item {item_name} has invalid small word durations: {item["word_durs"]}')
                    if i == 0:
                        item['word_durs'][i + 1] += wd
                    else:
                        word_durs[-1] += wd  # this word often be <SP>
                else:
                    word_durs.append(wd)
            mel2word, dur_word = align_word(word_durs, mel_len, self.hop_size, self.sr)
            if mel2word[0] == 0:  # better start from 1, consistent with mel2ph
                mel2word = [i + 1 for i in mel2word]
            mel2word_len = sum((mel2word > 0).astype(int))
            word_durs = torch.FloatTensor(word_durs[:hparams['max_input_tokens']])
        else:
            mel2word = word_durs = None
            mel2word_len = np.inf

        max_frames = hparams['max_frames']
        T = min(mel_len, mel2word_len)
        real_len = T
        T = math.ceil(min(T, max_frames) / hparams['frames_multiple']) * hparams['frames_multiple']
        # mel = pad_or_cut_xd(mel, T, dim=0)

        sample = {
            "id": idx,
            "item_name": item_name,
            # "mel": mel,
            # "mel_nonpadding": mel.abs().sum(-1) > 0,
            "real_len": real_len,
            "len": T,
            "wav": torch.from_numpy(wav),
        }
        # sample['mel_nonpadding'] = pad_or_cut_xd(sample['mel_nonpadding'].float(), T, 0)
        mel_nonpadding = torch.zeros(T)
        mel_nonpadding[:real_len] = 1.
        sample['mel_nonpadding'] = mel_nonpadding
        if mel2word is not None and word_durs is not None:
            sample['mel2word'] = pad_or_cut_xd(torch.LongTensor(mel2word), T, 0) if mel2word is not None else None
            sample["word_dur"] = word_durs
            sample["word_len"] = word_durs.shape[0]

        if 'word_durs' in item and mel2word is not None:
            sample["mel2word"] = mel2word = pad_or_cut_xd(torch.LongTensor(mel2word), T, 0)
            # make boundary labels for word
            word_bd = torch.zeros_like(mel2word)
            for idx in range(1, real_len):
                if mel2word[idx] != mel2word[idx - 1]:
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

        if 'mel' in sample:
            del sample['mel']

        return sample

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        batch = {
            'id': torch.LongTensor([s['id'] for s in samples]),
            'item_name': [s['item_name'] for s in samples],
            'nsamples': len(samples),
            'mel_nonpadding': collate_1d_or_2d([s['mel_nonpadding'] for s in samples], 0.0) if 'mel_nonpadding' in samples[0] else None,
            'real_lens': [s['real_len'] for s in samples],
            'lens': [s['len'] for s in samples],
            'wav': collate_1d_or_2d([s['wav'] for s in samples], 0.0) if 'wav' in samples[0] else None,
            'mel2word': collate_1d_or_2d([s['mel2word'] for s in samples], 0) if 'mel2word' in samples[0] else None,
            'word_durs': collate_1d_or_2d([s['word_dur'] for s in samples], 0.0) if 'word_dur' in samples[0] else None,
            'word_bd': collate_1d_or_2d([s['word_bd'] for s in samples], 0.0) if 'word_bd' in samples[0] else None,
            'word_lens': [s['word_len'] for s in samples] if 'word_len' in samples[0] else None,
        }
        return batch

    def __len__(self):
        return len(self.items)

    def ordered_indices(self):
        return (np.arange(len(self))).tolist()

    def num_tokens(self, idx):
        return self.sizes[idx]

def check_slur_cnt(note2words, item_name=None, verbose=False):
    cnt = 1
    slur_cnt = defaultdict(int)
    for note_idx in range(1, len(note2words)):
        if note2words[note_idx] == note2words[note_idx - 1]:
            cnt += 1
        else:
            if cnt > 1:
                if cnt > 2 and verbose:
                    print(f"warning: item [{item_name}] has {cnt} notes to 1 word.")
                slur_cnt[cnt] += 1
            cnt = 1
    return slur_cnt


if __name__ == '__main__':
    mp.set_start_method('spawn')
    rosvot = RosvotInfer()
    rosvot.run()
