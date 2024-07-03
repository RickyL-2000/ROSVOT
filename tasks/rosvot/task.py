import os
import sys
import traceback
from collections import defaultdict
import filecmp

import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.functional.classification import binary_auroc, binary_recall, binary_f1_score, binary_precision
import matplotlib.pyplot as plt
from tqdm import tqdm
import mir_eval
import pretty_midi
import glob

from utils import seed_everything
from utils.commons.hparams import hparams
from utils.commons.base_task import BaseTask
from utils.audio.pitch_utils import denorm_f0, boundary2Interval, midi_to_hz, save_midi, \
    validate_pitch_and_itv, midi_melody_eval, melody_eval_pitch_and_itv
from utils.commons.dataset_utils import data_loader, BaseConcatDataset, build_dataloader
from utils.commons.tensor_utils import tensors_to_scalars
from utils.commons.ckpt_utils import load_ckpt
from utils.nn.model_utils import print_arch
from utils.commons.multiprocess_utils import MultiprocessManager
from utils.audio.pitch_utils import midi_onset_eval, midi_offset_eval, midi_pitch_eval, \
    midi_COn_eval, midi_COnP_eval, midi_COnPOff_eval
from utils.commons.multiprocess_utils import multiprocess_run_tqdm
from utils.commons.losses import sigmoid_focal_loss
from utils.nn.schedulers import RSQRTSchedule, NoneSchedule, WarmupSchedule
# from utils.commons.gpu_mem_track import MemTracker

from tasks.rosvot.dataset import MidiDataset, WordbdDataset
from tasks.rosvot.rosvot_utils import bd_to_durs, bd_to_idxs, regulate_ill_slur, regulate_real_note_itv
from modules.rosvot.rosvot import MidiExtractor, WordbdExtractor

def parse_dataset_configs():
    max_tokens = hparams['max_tokens']
    max_sentences = hparams['max_sentences']
    max_valid_tokens = hparams['max_valid_tokens']
    if max_valid_tokens == -1:
        hparams['max_valid_tokens'] = max_valid_tokens = max_tokens
    max_valid_sentences = hparams['max_valid_sentences']
    if max_valid_sentences == -1:
        hparams['max_valid_sentences'] = max_valid_sentences = max_sentences
    return max_tokens, max_sentences, max_valid_tokens, max_valid_sentences

def f0_to_figure(f0_gt, f0_cwt=None, f0_pred=None):
    fig = plt.figure(figsize=(12, 8))
    f0_gt = f0_gt.cpu().numpy()
    plt.plot(f0_gt, color='r', label='gt')
    if f0_cwt is not None:
        f0_cwt = f0_cwt.cpu().numpy()
        plt.plot(f0_cwt, color='b', label='ref')
    if f0_pred is not None:
        f0_pred = f0_pred.cpu().numpy()
        plt.plot(f0_pred, color='green', label='pred')
    plt.legend()
    return fig

def f0_notes_to_figure(f0_gt, notes1, notes1_bd, label1='pred', notes2=None, notes2_bd=None, label2='other'):
    fig = plt.figure(figsize=(12, 8))
    plt.plot(f0_gt, color='r', label='gt f0')
    f0_1 = np.zeros(notes1_bd.shape[0])
    notes1_itv = boundary2Interval(notes1_bd)
    for i, itv in enumerate(notes1_itv):
        f0_1[itv[0]: itv[1]] = notes1[i]
    plt.plot(f0_1, color='green', label=label1)
    if notes2 is not None and notes2_bd is not None:
        f0_2 = np.zeros(notes2_bd.shape[0])
        notes2_itv = boundary2Interval(notes2_bd)
        for i, itv in enumerate(notes2_itv):
            f0_2[itv[0]: itv[1]] = notes2[i]
        plt.plot(f0_2, color='blue', label=label2)
    plt.legend()
    return fig

class MidiExtractorTask(BaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_cls = MidiDataset
        self.vocoder = None
        self.saving_result_pool = None
        self.saving_results_futures = None
        self.max_tokens, self.max_sentences, \
            self.max_valid_tokens, self.max_valid_sentences = parse_dataset_configs()
        seed_everything(hparams['seed'])

    @data_loader
    def train_dataloader(self):
        train_dataset = self.dataset_cls(prefix=hparams['train_set_name'], shuffle=True)
        return build_dataloader(train_dataset, True, self.max_tokens, self.max_sentences, endless=hparams['endless_ds'],
                                pin_memory=hparams.get('pin_memory', False), use_ddp=self.trainer.use_ddp)

    @data_loader
    def val_dataloader(self):
        valid_dataset = self.dataset_cls(prefix=hparams['valid_set_name'], shuffle=False)
        return build_dataloader(valid_dataset, False, self.max_valid_tokens, self.max_valid_sentences,
                                apply_batch_by_size=False, pin_memory=hparams.get('pin_memory', False),
                                use_ddp=self.trainer.use_ddp)

    @data_loader
    def test_dataloader(self):
        test_dataset = self.dataset_cls(prefix=hparams['test_set_name'], shuffle=False)
        self.test_dl = build_dataloader(
            test_dataset, False, self.max_valid_tokens, self.max_valid_sentences,
            apply_batch_by_size=False, pin_memory=hparams.get('pin_memory', False), use_ddp=self.trainer.use_ddp)
        return self.test_dl

    def build_model(self):
        self.model = MidiExtractor(hparams)
        if hparams['load_ckpt'] != '':
            load_ckpt(self.model, hparams['load_ckpt'])
        print_arch(self.model)
        return self.model

    def build_scheduler(self, optimizer):
        last_step = max(-1, self.global_step-hparams.get('accumulate_grad_batches', 1))
        if hparams['scheduler'] == 'rsqrt':
            return RSQRTSchedule(optimizer, hparams['lr'],
                                 round(hparams['warmup_updates'] / hparams.get('accumulate_grad_batches', 1)),
                                 hparams['hidden_size'], last_step=last_step)
        elif hparams['scheduler'] == 'warmup':
            return WarmupSchedule(optimizer, hparams['lr'],
                                  hparams['warmup_updates'] / hparams.get('accumulate_grad_batches', 1),
                                  last_step=last_step)
        elif hparams['scheduler'] == 'step_lr':
            return torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer,
                step_size=round(hparams.get('scheduler_lr_step_size', 500) / hparams.get('accumulate_grad_batches', 1)),
                gamma=hparams.get('scheduler_lr_gamma', 0.998), last_epoch=last_step)
        else:
            return NoneSchedule(optimizer, hparams['lr'])

    def build_optimizer(self, model):
        self.optimizer = optimizer = torch.optim.AdamW(
            [{'params': model.parameters(), 'initial_lr': hparams['lr']}],
            lr=hparams['lr'],
            betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
            weight_decay=hparams['weight_decay'])

        return optimizer

    def _training_step(self, sample, batch_idx, _):
        loss_output, _ = self.run_model(sample)
        total_loss = sum([v for v in loss_output.values() if isinstance(v, torch.Tensor) and v.requires_grad])
        loss_output['batch_size'] = sample['nsamples']
        return total_loss, loss_output

    def run_model(self, sample, infer=False):
        # gpu_tracker.track()
        mel = sample['mels']
        word_bd = sample['word_bd']
        notes = sample['notes']
        note_bd = sample['note_bd']     # [B, T]
        note_bd_soft = sample['note_bd_soft']
        pitch_coarse = sample['pitch_coarse']
        uv = sample['uv'].long()
        mel_nonpadding = sample['mel_nonpadding']

        output = self.model(mel=mel, word_bd=word_bd, note_bd=note_bd, pitch=pitch_coarse, uv=uv,
                            non_padding=mel_nonpadding, train=not infer)
        losses = {}
        if not infer:
            self.add_note_bd_loss(output['note_bd_logits'], note_bd_soft, losses, word_bd)
            try:
                self.add_note_pitch_loss(output['note_logits'], notes, losses)
            except RuntimeError as err:
                _, exc_value, exc_tb = sys.exc_info()
                tb = traceback.extract_tb(exc_tb)[-1]
                print(f'skip {sample["item_name"][:4]}{"..." if len(sample["item_name"]) > 4 else ""}, '
                      f'{err}: {exc_value} in {tb[0]}:{tb[1]} "{tb[2]}" in {tb[3]}')
                losses['pitch'] = 0.0
        # gpu_tracker.track()
        return losses, output

    def add_note_bd_loss(self, note_bd_logits, note_bd_soft, losses, word_bd=None):
        if self.global_step >= hparams.get('note_bd_start', 0):
            if not hasattr(self, 'note_bd_pos_weight'):
                self.note_bd_pos_weight = torch.ones(5000).to(note_bd_logits.device)    # cache
                if hparams.get('label_pos_weight_decay', 0.0) > 0.0:
                    note_bd_ratio = hparams.get('note_bd_ratio', 3) * hparams['hop_size'] / hparams['audio_sample_rate']
                    note_bd_pos_weight = 1 / note_bd_ratio
                    self.note_bd_pos_weight = self.note_bd_pos_weight * note_bd_pos_weight * hparams.get('label_pos_weight_decay', 0.0)
            note_bd_pos_weight = self.note_bd_pos_weight[:note_bd_logits.shape[1]]
            note_bd_loss = F.binary_cross_entropy_with_logits(note_bd_logits, note_bd_soft, pos_weight=note_bd_pos_weight)
            losses['note_bd'] = note_bd_loss * hparams.get('lambda_note_bd', 1.0)
            # add slur punish.
            if hparams.get('lambda_note_bd_slur_punish', 0.0) > 0 and word_bd is not None:
                slur_punish = torch.sigmoid(note_bd_logits) * word_bd.eq(0).float()
                slur_punish = torch.sum(slur_punish, dim=-1) / (torch.sum(word_bd.eq(0).float(), dim=-1) + 1e-5)
                losses['slur_punish'] = torch.mean(slur_punish) * hparams.get('lambda_note_bd_slur_punish', 0.0)
            # add focal loss
            if hparams.get('note_bd_focal_loss', None) not in ['none', None, 0]:
                gamma = float(hparams.get('note_bd_focal_loss', None))
                focal_loss = sigmoid_focal_loss(
                    note_bd_logits, note_bd_soft, alpha=1 / self.note_bd_pos_weight[0], gamma=gamma, reduction='mean')
                losses['note_bd_fc'] = focal_loss * hparams.get('lambda_note_bd_focal', 1.0)
        else:
            losses['note_bd'] = 40.0 * hparams.get('lambda_note_bd', 1.0)
            if hparams.get('note_bd_focal_loss', None) not in ['none', None]:
                losses['note_bd_fc'] = 40.0 * hparams.get('lambda_note_bd_focal', 1.0)

    def add_note_pitch_loss(self, note_logits, notes, losses):
        if self.global_step >= hparams.get('note_pitch_start', 0):
            note_pitch_loss = F.cross_entropy(note_logits.transpose(1, 2), notes,
                                              label_smoothing=hparams.get('note_pitch_label_smoothing', 0.0))
            losses['pitch'] = note_pitch_loss * hparams.get('lambda_note_pitch', 1.0)
        else:
            losses['pitch'] = 40.0

    def validation_start(self):
        pass

    def validation_step(self, sample, batch_idx):
        outputs = {}
        outputs['losses'] = {}
        with torch.no_grad():
            outputs['losses'], model_out = self.run_model(sample, infer=False)
        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs['nsamples'] = sample['nsamples']
        if batch_idx < hparams['num_valid_stats']:
            mel = sample['mels']
            word_bd = sample['word_bd']
            notes = sample['notes']
            note_bd = sample['note_bd']
            pitch_coarse = sample['pitch_coarse']
            uv = sample['uv'].long()
            mel_nonpadding = sample['mel_nonpadding']
            gt_f0 = denorm_f0(sample['f0'], sample["uv"])

            with torch.no_grad():
                # no note_bd
                output_3 = self.model(mel=mel, word_bd=word_bd, note_bd=None, pitch=pitch_coarse, uv=uv,
                                      non_padding=mel_nonpadding, train=True)
                note_bd_pred = output_3['note_bd_pred'][0].data.cpu().numpy()  # [B, T] or [1, T] -> [T]
                note_bd_gt = note_bd[0].data.cpu().numpy()
                note_itv_gt = boundary2Interval(note_bd_gt) * hparams['hop_size'] / hparams['audio_sample_rate']
                note_itv_pred = boundary2Interval(note_bd_pred) * hparams['hop_size'] / hparams['audio_sample_rate']
                note_gt = notes[0]  # [B, note_length] -> [note_length]
                note_gt = midi_to_hz(note_gt.data.cpu().numpy())
                note_pred = output_3['note_pred'][0]
                note_pred = midi_to_hz(note_pred.data.cpu().numpy())
                if batch_idx < hparams['num_valid_plots']:
                    self.logger.add_figure(
                        f'note_{batch_idx}',
                        f0_notes_to_figure(gt_f0[0].data.cpu().numpy(), note_gt,
                                           note_bd_gt, 'gt note', note_pred,
                                           note_bd_pred, 'pred note'),
                        self.global_step)
                try:
                    note_gt, note_itv_gt = validate_pitch_and_itv(note_gt, note_itv_gt)
                    note_pred, note_itv_pred = validate_pitch_and_itv(note_pred, note_itv_pred)
                    note_onset_p, note_onset_r, note_onset_f = mir_eval.transcription.onset_precision_recall_f1(
                        note_itv_gt, note_itv_pred, onset_tolerance=0.1, strict=False, beta=1.0)
                    if note_itv_gt.shape == (0,) or note_itv_pred.shape == (0,):
                        overlap_p, overlap_r, overlap_f, avg_overlap_ratio = 0, 0, 0, 0
                    else:
                        overlap_p, overlap_r, overlap_f, avg_overlap_ratio = mir_eval.transcription.precision_recall_f1_overlap(
                            note_itv_gt, note_gt, note_itv_pred, note_pred, onset_tolerance=0.1, pitch_tolerance=50.0,
                            offset_ratio=0.2, offset_min_tolerance=0.05, strict=False, beta=1.0)
                    vr, vfa, rpa, rca, oa = melody_eval_pitch_and_itv(
                        note_gt, note_itv_gt, note_pred, note_itv_pred, hparams['hop_size'], hparams['audio_sample_rate'])
                except Exception as err:
                    note_onset_p, note_onset_r, note_onset_f = 0, 0, 0
                    overlap_p, overlap_r, overlap_f, avg_overlap_ratio = 0, 0, 0, 0
                    vr, vfa, rpa, rca, oa = 0, 0, 0, 0, 0
                    _, exc_value, exc_tb = sys.exc_info()
                    tb = traceback.extract_tb(exc_tb)[-1]
                    print(f'skip {sample["item_name"][:2]} {"..." if len(sample["item_name"]) > 2 else ""}, '
                          f'{err}: {exc_value} in {tb[0]}:{tb[1]} "{tb[2]}" in {tb[3]}')
                outputs['losses']['COn_p'], outputs['losses']['COn_r'], outputs['losses']['COn_f'] = \
                    note_onset_p, note_onset_r, note_onset_f
                outputs['losses']['COnPOff_p'], outputs['losses']['COnPOff_r'], outputs['losses']['COnPOff_f'], \
                    outputs['losses']['aor'] = overlap_p, overlap_r, overlap_f, avg_overlap_ratio
                outputs['losses']['vr'], outputs['losses']['vfa'], outputs['losses']['rpa'], outputs['losses']['oa'] = \
                    vr, vfa, rpa, oa
                if batch_idx < hparams['num_valid_plots']:
                    self.logger.add_figure(
                        f'note_w_gt_word_bd_{batch_idx}',
                        f0_notes_to_figure(gt_f0[0].data.cpu().numpy(), np.cumsum(np.ones(note_bd_gt.shape[0] + 1) * 40),
                                           note_bd_gt, 'gt note', np.cumsum(np.ones(note_bd_gt.shape[0] + 1) * 50),
                                           note_bd_pred, 'pred note'),
                        self.global_step)

            self.save_valid_result(sample, batch_idx, model_out)
        outputs = tensors_to_scalars(outputs)
        return outputs

    def validation_end(self, outputs):
        # torch.cuda.empty_cache()
        return super(MidiExtractorTask, self).validation_end(outputs)

    def save_valid_result(self, sample, batch_idx, model_out):
        pass

    def test_start(self):
        self.saving_result_pool = MultiprocessManager(int(os.getenv('N_PROC', os.cpu_count())))
        self.saving_results_futures = []
        self.gen_dir = os.path.join(
            hparams['work_dir'], f'generated_{self.trainer.global_step}_{hparams["gen_dir_name"]}')
        os.makedirs(self.gen_dir, exist_ok=True)
        os.makedirs(f'{self.gen_dir}/plot', exist_ok=True)
        os.makedirs(f'{self.gen_dir}/midi', exist_ok=True)

    def test_step(self, sample, batch_idx):
        _, outputs = self.run_model(sample, infer=True)
        f0 = denorm_f0(sample['f0'], sample['uv'])[0].cpu().numpy()
        note_bd_pred = outputs['note_bd_pred'][0].cpu().numpy()
        note_pred = outputs['note_pred'][0].cpu().numpy()
        note_bd_prob = torch.sigmoid(outputs['note_bd_logits'])[0].cpu().numpy()
        note_bd_gt = sample['note_bd'][0].cpu().numpy()
        note_gt = sample['notes'][0].cpu().numpy()
        note_durs_gt = sample['note_durs'][0].cpu().numpy()
        word_bd = sample['word_bd'][0].cpu().numpy()
        word_durs = sample['word_durs'][0].cpu().numpy()
        item_name = sample['item_name'][0]
        # mel_gt = sample['mels'][0].cpu().numpy()
        base_fn = f'{item_name}[%s]'
        base_fn = base_fn.replace(' ', '_')
        gen_dir = self.gen_dir
        self.saving_result_pool.add_job(self.save_result, args=[
            base_fn, gen_dir, f0, note_bd_pred, note_pred, note_bd_gt, note_gt, note_durs_gt, note_bd_prob, word_bd, word_durs])
        return {}

    @staticmethod
    def save_result(base_fn, gen_dir, gt_f0=None, note_bd_pred=None, note_pred=None,
                    note_bd_gt=None, note_gt=None, note_durs_gt=None, note_bd_prob=None, word_bd=None, word_durs=None):
        if note_pred.shape == (0,):
            print(f"skip {base_fn % ''}: no notes detected")
            return
        fn = base_fn % 'P'
        note_itv_pred = boundary2Interval(note_bd_pred)
        note_itv_gt = boundary2Interval(note_bd_gt)
        if hparams.get('infer_regulate_real_note_itv', True) and hparams.get('use_wbd', True):
            note_itv_pred_secs, note2words = regulate_real_note_itv(note_itv_pred, note_bd_pred, word_bd, word_durs, hparams['hop_size'], hparams['audio_sample_rate'])
            note_pred, note_itv_pred_secs, note2words = regulate_ill_slur(note_pred, note_itv_pred_secs, note2words)
            save_midi(note_pred, note_itv_pred_secs, f'{gen_dir}/midi/{fn}.mid')
            # 为防止舍入误差，使用 gt 的 dur
            if hparams['save_gt']:
                note_itv_gt_secs = np.zeros((note_durs_gt.shape[0], 2))
                note_offsets = np.cumsum(note_durs_gt)
                for idx in range(len(note_offsets) - 1):
                    note_itv_gt_secs[idx, 1] = note_itv_gt_secs[idx + 1, 0] = note_offsets[idx]
                note_itv_gt_secs[-1, 1] = note_offsets[-1]
                fn = base_fn % 'G'
                save_midi(note_gt, note_itv_gt_secs, f'{gen_dir}/midi/{fn}.mid')
        else:
            note_itv_pred_secs = note_itv_pred * hparams['hop_size'] / hparams['audio_sample_rate']
            save_midi(note_pred, note_itv_pred_secs, f'{gen_dir}/midi/{fn}.mid')
            if hparams['save_gt']:
                fn = base_fn % 'G'
                save_midi(note_gt, note_itv_gt * hparams['hop_size'] / hparams['audio_sample_rate'],
                          f'{gen_dir}/midi/{fn}.mid')

        fn = base_fn % ''
        fig = plt.figure()
        plt.plot(gt_f0, color='red', label='gt f0')
        midi_pred = np.zeros(note_bd_pred.shape[0])
        for i, itv in enumerate(np.round(note_itv_pred_secs * hparams['audio_sample_rate'] / hparams['hop_size']).astype(int)):
            midi_pred[itv[0]: itv[1]] = note_pred[i]
        midi_pred = midi_to_hz(midi_pred)
        plt.plot(midi_pred, color='blue', label='pred midi')
        midi_gt = np.zeros(note_bd_gt.shape[0])
        for i, itv in enumerate(note_itv_gt):
            midi_gt[itv[0]: itv[1]] = note_gt[i]
        midi_gt = midi_to_hz(midi_gt)
        plt.plot(midi_gt, color='yellow', label='gt midi')
        plt.plot(note_bd_prob * 100, color='green', label='note bd prob')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{gen_dir}/plot/[F0][{fn}].png', format='png')
        plt.close(fig)

    def test_end(self, outputs):
        path_list = sorted(glob.glob(f"{self.gen_dir}/midi/*.mid"))
        pair_list = []
        for path in path_list:
            if '[G]' in path:
                gt_path = path
                pred_path = gt_path.replace('[G]', '[P]')
                if str(pred_path) not in set(path_list):
                    print(f'skip {pred_path}: no such file')
                    continue
                pair_list.append([str(gt_path), str(pred_path)])

        COn_scores = np.zeros(3)
        COnP_scores = np.zeros(4)
        COnPOff_scores = np.zeros(4)
        melody_scores = np.zeros(5)
        for item_id, (COn_scores_, COnP_scores_, COnPOff_scores_, melody_scores_) in \
                multiprocess_run_tqdm(evaluate, pair_list, desc='evaluating'):
            COn_scores += np.array(COn_scores_)
            COnP_scores += np.array(COnP_scores_)
            COnPOff_scores += np.array(COnPOff_scores_)
            melody_scores += np.array(melody_scores_)

        COn_scores /= len(pair_list)
        COnP_scores /= len(pair_list)
        COnPOff_scores /= len(pair_list)
        melody_scores /= len(pair_list)

        print(f'| TL;DR: COn: {COn_scores[2]:.3f}, COnP: {COnP_scores[2]:.3f}, COnPOff: {COnPOff_scores[2]:.3f}')
        print(f'| Metrics |     f1    | precision |   recall   |')
        print(f'|   COn   |   {COn_scores[2]:.3f}   |   {COn_scores[0]:.3f}   |    {COn_scores[1]:.3f}   |')
        print(f'|   COnP  |   {COnP_scores[2]:.3f}   |   {COnP_scores[0]:.3f}   |    {COnP_scores[1]:.3f}   |')
        print(f'| COnPOff |   {COnPOff_scores[2]:.3f}   |   {COnPOff_scores[0]:.3f}   |    {COnPOff_scores[1]:.3f}   |')
        print(f'| melody: VR: {melody_scores[0]:.3f}, VFA: {melody_scores[1]:.3f}, RPA: {melody_scores[2]:.3f}, RCA: {melody_scores[3]:.3f}, OA: {melody_scores[4]:.3f}')

        return {}

def evaluate(mid_gt_path, mid_pred_path):
    mid_true = pretty_midi.PrettyMIDI(mid_gt_path)
    mid_pred = pretty_midi.PrettyMIDI(mid_pred_path)

    COn_scores = midi_COn_eval(mid_true, mid_pred)
    COnP_scores = midi_COnP_eval(mid_true, mid_pred)
    COnPOff_scores = midi_COnPOff_eval(mid_true, mid_pred)
    melody_scores = midi_melody_eval(mid_true, mid_pred, hop_size=256, sample_rate=48000)

    return COn_scores, COnP_scores, COnPOff_scores, melody_scores

class RobustWordbdTask(MidiExtractorTask):
    def __init__(self, *args, **kwargs):
        super(RobustWordbdTask, self).__init__(*args, **kwargs)
        self.dataset_cls = WordbdDataset

    def build_model(self):
        self.model = WordbdExtractor(hparams)
        if hparams['load_ckpt'] != '':
            load_ckpt(self.model, hparams['load_ckpt'])
        print_arch(self.model)
        return self.model

    def run_model(self, sample, infer=False):
        mel = sample['mels']
        word_bd = sample['word_bd']
        word_bd_soft = sample['word_bd_soft']
        pitch_coarse = sample['pitch_coarse']
        uv = sample['uv'].long()
        mel_nonpadding = sample['mel_nonpadding']

        output = self.model(mel=mel, pitch=pitch_coarse, uv=uv, non_padding=mel_nonpadding, train=not infer)
        losses = {}
        if not infer:
            self.add_word_bd_loss(output['word_bd_logits'], word_bd_soft, losses)

        return losses, output

    def add_word_bd_loss(self, word_bd_logits, word_bd_soft, losses):
        if self.global_step >= hparams.get('word_bd_start', 0):
            if not hasattr(self, 'word_bd_pos_weight'):
                self.word_bd_pos_weight = torch.ones(hparams['max_frames']).to(word_bd_logits.device)    # cache
                if hparams.get('label_pos_weight_decay', 0.0) > 0.0:
                    word_bd_ratio = hparams.get('word_bd_ratio', 2) * hparams['hop_size'] / hparams['audio_sample_rate']
                    if hparams.get('use_soft_word_bd', False) and hparams.get('soft_word_bd_func', None) is not None:
                        soft_label_func, win_size = hparams.get('soft_word_bd_func', None).split(':')
                        win_size = round(int(win_size) * hparams['audio_sample_rate'] / 1000 / hparams['hop_size'])
                        win_size = win_size if win_size % 2 == 1 else win_size + 1  # ensure odd number
                        word_bd_ratio = word_bd_ratio * win_size / 3  # only use the middle 1/3 counted as a label
                    word_bd_pos_weight = 1 / word_bd_ratio
                    self.word_bd_pos_weight = self.word_bd_pos_weight * word_bd_pos_weight * hparams.get('label_pos_weight_decay', 0.0)
            word_bd_pos_weight = self.word_bd_pos_weight[:word_bd_logits.shape[1]]
            word_bd_loss = F.binary_cross_entropy_with_logits(word_bd_logits, word_bd_soft, pos_weight=word_bd_pos_weight)
            losses['word_bd'] = word_bd_loss * hparams.get('lambda_word_bd', 1.0)
            # add focal loss
            if hparams.get('word_bd_focal_loss', None) not in ['none', None, 0]:
                gamma = float(hparams.get('word_bd_focal_loss', None))
                focal_loss = sigmoid_focal_loss(
                    word_bd_logits, word_bd_soft, alpha=1 / self.word_bd_pos_weight[0], gamma=gamma, reduction='mean')
                losses['word_bd_fc'] = focal_loss * hparams.get('lambda_word_bd_focal', 1.0)
        else:
            losses['word_bd'] = 40.0 * hparams.get('lambda_word_bd', 1.0)
            if hparams.get('word_bd_focal_loss', None) not in ['none', None]:
                losses['word_bd_fc'] = 40.0 * hparams.get('lambda_word_bd_focal', 1.0)

    def validation_step(self, sample, batch_idx):
        outputs = {}
        outputs['losses'] = {}
        with torch.no_grad():
            outputs['losses'], model_out = self.run_model(sample, infer=False)
        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs['nsamples'] = sample['nsamples']
        if batch_idx < hparams['num_valid_stats']:
            mel = sample['mels']
            word_bd = sample['word_bd']
            word_bd_soft = sample['word_bd_soft']
            pitch_coarse = sample['pitch_coarse']
            uv = sample['uv'].long()
            mel_nonpadding = sample['mel_nonpadding']
            gt_f0 = denorm_f0(sample['f0'], sample["uv"])

            with torch.no_grad():
                output = self.model(mel=mel, pitch=pitch_coarse, uv=uv, non_padding=mel_nonpadding, train=False)

                outputs['losses']['swbd_auroc'] = binary_auroc(output['word_bd_logits'], word_bd,
                                                               hparams.get('word_bd_threshold', 0.9))
                outputs['losses']['swbd_p'] = binary_precision(output['word_bd_logits'], word_bd,
                                                               hparams.get('word_bd_threshold', 0.9))
                outputs['losses']['wbd_p'] = binary_precision(output['word_bd_pred'], word_bd)
                outputs['losses']['swbd_r'] = binary_recall(output['word_bd_logits'], word_bd,
                                                            hparams.get('word_bd_threshold', 0.9))
                outputs['losses']['wbd_r'] = binary_recall(output['word_bd_pred'], word_bd)
                outputs['losses']['swbd_f'] = binary_f1_score(output['word_bd_logits'], word_bd,
                                                              hparams.get('word_bd_threshold', 0.9))
                outputs['losses']['wbd_f'] = binary_f1_score(output['word_bd_pred'], word_bd)

                word_bd_gt = word_bd[0].data.cpu().numpy()
                word_bd_pred = output['word_bd_pred'][0].data.cpu().numpy()
                word_bd_prob = torch.sigmoid(output['word_bd_logits'])[0].data.cpu().numpy()
                if batch_idx < hparams['num_valid_plots']:
                    self.logger.add_figure(
                        f'word_bd_{batch_idx}',
                        f0_words_to_figure(gt_f0[0].data.cpu().numpy(), np.cumsum(np.ones(word_bd_gt.shape[0] + 1) * 40),
                                           word_bd_gt, 'gt wbd', np.cumsum(np.ones(word_bd_gt.shape[0] + 1) * 50),
                                           word_bd_pred, 'pred wbd', word_bd_prob),
                        self.global_step)

        outputs = tensors_to_scalars(outputs)
        return outputs

    def test_start(self):
        self.saving_result_pool = MultiprocessManager(int(os.getenv('N_PROC', os.cpu_count())))
        self.saving_results_futures = []
        self.gen_dir = os.path.join(
            hparams['work_dir'], f'generated_{self.trainer.global_step}_{hparams["gen_dir_name"]}')
        os.makedirs(self.gen_dir, exist_ok=True)
        os.makedirs(f'{self.gen_dir}/plot', exist_ok=True)

    def test_step(self, sample, batch_idx):
        _, outputs = self.run_model(sample, infer=True)
        f0 = denorm_f0(sample['f0'], sample['uv'])[0].cpu().numpy()
        word_bd_gt = sample['word_bd'][0].cpu().numpy()
        word_bd_pred = outputs['word_bd_pred'][0].cpu().numpy()
        word_bd_prob = torch.sigmoid(outputs['word_bd_logits'])[0].data.cpu().numpy()
        word_bd_logits = outputs['word_bd_logits'][0].data.cpu().numpy()
        item_name = sample['item_name'][0]
        base_fn = f'{item_name}[%s]'
        base_fn = base_fn.replace(' ', '_')
        gen_dir = self.gen_dir
        self.saving_result_pool.add_job(self.save_result, args=[
            base_fn, gen_dir, f0, word_bd_pred, word_bd_prob, word_bd_logits, word_bd_gt, hparams.get('word_bd_threshold', 0.9)])
        return {}

    @staticmethod
    def save_result(base_fn, gen_dir, gt_f0=None, word_bd_pred=None, word_bd_prob=None, word_bd_logits=None,
                    word_bd_gt=None, word_bd_threshold=None):
        if word_bd_pred.sum() == 0:
            print(f"skip {base_fn % ''}: no words detected")
            word_bd_pred = None
            # return {}
        fn = base_fn % ''
        fig = f0_words_to_figure(gt_f0[0], np.cumsum(np.ones(word_bd_gt.shape[0] + 1) * 40),
                                 word_bd_gt, 'gt wbd', np.cumsum(np.ones(word_bd_gt.shape[0] + 1) * 50),
                                 word_bd_pred, 'pred wbd', word_bd_prob, save_path=f'{gen_dir}/plot/[wordbd][{fn}].png')
        plt.close(fig)

        # TODO: develop a soft evaluation
        res = {}
        res['swbd_auroc'] = binary_auroc(torch.Tensor(word_bd_logits), torch.LongTensor(word_bd_gt), word_bd_threshold)
        res['swbd_p'] = binary_precision(torch.Tensor(word_bd_logits), torch.LongTensor(word_bd_gt), word_bd_threshold)
        res['swbd_r'] = binary_recall(torch.Tensor(word_bd_logits), torch.LongTensor(word_bd_gt), word_bd_threshold)
        res['swbd_f'] = binary_f1_score(torch.Tensor(word_bd_logits), torch.LongTensor(word_bd_gt), word_bd_threshold)
        if word_bd_pred is not None:
            res['wbd_p'] = binary_precision(torch.Tensor(word_bd_pred), torch.LongTensor(word_bd_gt))
            res['wbd_r'] = binary_recall(torch.Tensor(word_bd_pred), torch.LongTensor(word_bd_gt))
            res['wbd_f'] = binary_f1_score(torch.Tensor(word_bd_pred), torch.LongTensor(word_bd_gt))

        return res

    def test_end(self, outputs):
        res = defaultdict(list)
        for r_id, r in tqdm(self.saving_result_pool.get_results(), total=len(self.saving_result_pool)):
            for k in r:
                res[k].append(r[k])

        print('| Evaluation:')
        avg_score = {}
        for k in res:
            avg_score[k] = np.mean(res[k])
            print(f"{k}: {avg_score[k]:.3f} ", end='')
        print()

        return {}

def f0_words_to_figure(f0_gt, notes1, notes1_bd, label1='pred', notes2=None, notes2_bd=None, label2='other',
                       word_bd_prob=None, save_path=None):
    fig = plt.figure(figsize=(12, 8))
    plt.plot(f0_gt, color='r', label='gt f0')
    f0_1 = np.zeros(notes1_bd.shape[0])
    notes1_itv = boundary2Interval(notes1_bd)
    for i, itv in enumerate(notes1_itv):
        f0_1[itv[0]: itv[1]] = notes1[i]
    plt.plot(f0_1, color='green', label=label1)
    if notes2 is not None and notes2_bd is not None:
        f0_2 = np.zeros(notes2_bd.shape[0])
        notes2_itv = boundary2Interval(notes2_bd)
        for i, itv in enumerate(notes2_itv):
            f0_2[itv[0]: itv[1]] = notes2[i]
        plt.plot(f0_2, color='blue', label=label2)
    if word_bd_prob is not None:
        plt.plot(word_bd_prob * 100, color='orange', label='logits')
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path, format='png')
    return fig
