import math
import os

from tqdm import tqdm
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.multiprocessing as mp
from torch.distributed import init_process_group
import torch.distributed as dist

from modules.pe.rmvpe import RMVPE
from utils.commons.dataset_utils import batch_by_size, build_dataloader
# import utils
from utils.audio import get_wav_num_frames

"""
A convenient API for batch inference
update: add ddp
"""

class RMVPEInferDataset(Dataset):
    def __init__(self, wav_fns: list, id_and_sizes=None, sr=24000, hop_size=128, num_workers=0):
        if id_and_sizes is None:
            id_and_sizes = []
            if type(wav_fns[0]) == str:  # wav_paths
                for idx, wav_path in enumerate(wav_fns):
                    total_frames = get_wav_num_frames(wav_path, sr)
                    id_and_sizes.append((idx, round(total_frames / hop_size)))
            else:  # numpy arrays, mono wavs
                for idx, wav in enumerate(wav_fns):
                    id_and_sizes.append((idx, round(wav.shape[-1] / hop_size)))
        self.wav_fns = wav_fns
        self.id_and_sizes = id_and_sizes
        self.sr = sr
        self.num_workers = num_workers

    def __getitem__(self, idx):
        if type(self.wav_fns[idx]) == str:
            wav_fn = self.wav_fns[idx]
            wav, _ = librosa.core.load(wav_fn, sr=self.sr)
        else:
            wav = self.wav_fns[idx]
        return idx, wav

    def collater(self, samples: list):
        return samples

    def __len__(self):
        return len(self.wav_fns)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        return np.arange(len(self))

    def num_tokens(self, index):
        return self.id_and_sizes[index][1]

@torch.no_grad()
def extract(wav_fns: list, id_and_sizes=None, ckpt=None, sr=24000, hop_size=128, bsz=128, max_tokens=100000,
             fmax=900, fmin=50, ds_workers=0):
    all_gpu_ids = [int(x) for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x != '']
    num_gpus = len(all_gpu_ids)
    dist_config = {
        "dist_backend": "nccl",
        "dist_url": "tcp://localhost:54189",
        "world_size": 1
    }
    # https://discuss.pytorch.org/t/how-to-fix-a-sigsegv-in-pytorch-when-using-distributed-training-e-g-ddp/113518/10#:~:text=Using%20start%20and%20join%20avoids
    # https://github.com/pytorch/pytorch/issues/40403#issuecomment-648515174
    # mp.set_start_method('spawn')
    if num_gpus > 1:
        result_queue = mp.Queue()
        for rank in range(num_gpus):
            mp.Process(target=extract_worker, args=(rank, wav_fns, id_and_sizes, ckpt, sr, hop_size, bsz, max_tokens, fmax,
                                                    fmin, dist_config, num_gpus, ds_workers, result_queue,)).start()
        f0_res = [None] * len(wav_fns)
        for _ in range(num_gpus):
            f0_res_dict = result_queue.get()
            for idx in f0_res_dict:
                f0_res[idx] = f0_res_dict[idx]
            del f0_res_dict
    else:
        # f0_res = extract_one_process(wav_fns, id_and_sizes, ckpt, sr, hop_size, bsz, max_tokens, fmax, fmin)
        f0_res_dict = extract_worker(0, wav_fns, id_and_sizes, ckpt, sr, hop_size, bsz, max_tokens, fmax,
                                     fmin, dist_config, num_gpus, ds_workers, None)
        f0_res = [None] * len(wav_fns)
        for idx in f0_res_dict:
            f0_res[idx] = f0_res_dict[idx]
    return f0_res

@torch.no_grad()
def extract_worker(rank, wav_fns: list, id_and_sizes=None, ckpt=None, sr=24000, hop_size=128, bsz=128, max_tokens=100000,
                   fmax=900, fmin=50, dist_config=None, num_gpus=1, ds_workers=0, q=None):
    # print(f"rank: {rank}")
    if num_gpus > 1:
        init_process_group(backend=dist_config['dist_backend'], init_method=dist_config['dist_url'],
                           world_size=dist_config['world_size'] * num_gpus, rank=rank)
    dataset = RMVPEInferDataset(wav_fns, id_and_sizes, sr, hop_size, num_workers=ds_workers)
    # ds_sampler = DistributedSampler(dataset, shuffle=False) if num_gpus > 1 else None
    # loader = DataLoader(dataset, sampler=ds_sampler, collate_fn=dataset.collator, batch_size=1, num_workers=40, drop_last=False)
    loader = build_dataloader(dataset, shuffle=False, max_tokens=max_tokens, max_sentences=bsz, use_ddp=num_gpus > 1)
    loader = tqdm(loader, desc=f'| Processing f0 in [n_ranks={num_gpus}; max_tokens={max_tokens}; max_sentences={bsz}]') if rank == 0 else loader

    device = torch.device(f"cuda:{int(rank)}")
    model = RMVPE(ckpt, device=device)
    f0_res_dict = {}
    for batch in loader:
        if batch is None or len(batch) == 0:
            continue
        idxs = [item[0] for item in batch]
        wavs = [item[1] for item in batch]
        lengths = [(wav.shape[0] + hop_size - 1) // hop_size for wav in wavs]
        with torch.no_grad():
            f0s, uvs = model.get_pitch_batch(
                wavs, sample_rate=sr,
                hop_size=hop_size,
                lengths=lengths,
                fmax=fmax,
                fmin=fmin
            )
        for i, idx in enumerate(idxs):
            f0_res_dict[idx] = f0s[i]
    if q is not None:
        q.put(f0_res_dict)
    else:
        return f0_res_dict

# old version
def extract_one_process(wav_fns: list, id_and_sizes=None, ckpt=None, sr=24000, hop_size=128, bsz=128, max_tokens=100000,
             fmax=900, fmin=50, device='cuda'):
    assert ckpt is not None
    rmvpe = RMVPE(ckpt, device=device)
    if id_and_sizes is None:
        id_and_sizes = []
        if type(wav_fns[0]) == str:    # wav_paths
            for idx, wav_path in enumerate(wav_fns):
                total_frames = get_wav_num_frames(wav_path, sr)
                id_and_sizes.append((idx, round(total_frames / hop_size)))
        else:                       # numpy arrays, mono wavs
            for idx, wav in enumerate(wav_fns):
                id_and_sizes.append((idx, round(wav.shape[-1] / hop_size)))
    get_size = lambda x: x[1]
    bs = batch_by_size(id_and_sizes, get_size, max_tokens=max_tokens, max_sentences=bsz)
    for i in range(len(bs)):
        bs[i] = [bs[i][j][0] for j in range(len(bs[i]))]

    f0_res = [None] * len(wav_fns)
    for batch in tqdm(bs, total=len(bs), desc=f'| Processing f0 in [max_tokens={max_tokens}; max_sentences={bsz}]'):
        wavs, mel_lengths, lengths = [], [], []
        for idx in batch:
            if type(wav_fns[idx]) == str:
                wav_fn = wav_fns[idx]
                wav, _ = librosa.core.load(wav_fn, sr=sr)
            else:
                wav = wav_fns[idx]
            wavs.append(wav)
            mel_lengths.append(math.ceil((wav.shape[0] + 1) / hop_size))
            lengths.append((wav.shape[0] + hop_size - 1) // hop_size)

        with torch.no_grad():
            f0s, uvs = rmvpe.get_pitch_batch(
                wavs, sample_rate=sr,
                hop_size=hop_size,
                lengths=lengths,
                fmax=fmax,
                fmin=fmin
            )

        for i, idx in enumerate(batch):
            f0_res[idx] = f0s[i]

    if rmvpe is not None:
        rmvpe.release_cuda()
        torch.cuda.empty_cache()
        rmvpe = None

    return f0_res



