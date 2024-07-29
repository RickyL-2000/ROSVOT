from copy import deepcopy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.commons.hparams import hparams
from utils.commons.gpu_mem_track import MemTracker
from modules.commons.layers import Embedding
from modules.commons.conv import ResidualBlock, ConvBlocks
from modules.commons.conformer.conformer import ConformerLayers
from modules.rosvot.unet import Unet

def regulate_boundary(bd_logits, threshold, min_gap=18, ref_bd=None, ref_bd_min_gap=8, non_padding=None):
    # this doesn't preserve gradient
    device = bd_logits.device
    bd_logits = torch.sigmoid(bd_logits).data.cpu()
    # bd_logits[0] = bd_logits[-1] = 1e-5     # avoid itv invalid problem
    bd = (bd_logits > threshold).long()
    bd_res = torch.zeros_like(bd).long()
    for i in range(bd.shape[0]):
        bd_i = bd[i]
        last_bd_idx = -1
        start = -1
        for j in range(bd_i.shape[0]):
            if bd_i[j] == 1:
                if 0 <= start < j:
                    continue
                elif start < 0:
                    start = j
            else:
                if 0 <= start < j:
                    if j - 1 > start:
                        bd_idx = start + int(torch.argmax(bd_logits[i, start: j]).item())
                    else:
                        bd_idx = start
                    if bd_idx - last_bd_idx < min_gap and last_bd_idx > 0:
                        bd_idx = round((bd_idx + last_bd_idx) / 2)
                        bd_res[i, last_bd_idx] = 0
                    bd_res[i, bd_idx] = 1
                    last_bd_idx = bd_idx
                    start = -1

    # assert ref_bd_min_gap <= min_gap // 2
    if ref_bd is not None and ref_bd_min_gap > 0:
        ref = ref_bd.data.cpu()
        for i in range(bd_res.shape[0]):
            ref_bd_i = ref[i]
            ref_bd_i_js = []
            for j in range(ref_bd_i.shape[0]):
                if ref_bd_i[j] == 1:
                    ref_bd_i_js.append(j)
                    seg_sum = torch.sum(bd_res[i, max(0, j - ref_bd_min_gap): j + ref_bd_min_gap])
                    if seg_sum == 0:
                        bd_res[i, j] = 1
                    elif seg_sum == 1 and bd_res[i, j] != 1:
                        bd_res[i, max(0, j - ref_bd_min_gap): j + ref_bd_min_gap] = \
                            ref_bd_i[max(0, j - ref_bd_min_gap): j + ref_bd_min_gap]
                    elif seg_sum > 1:
                        for k in range(1, ref_bd_min_gap+1):
                            if bd_res[i, max(0, j - k)] == 1 and ref_bd_i[max(0, j - k)] != 1:
                                bd_res[i, max(0, j - k)] = 0
                                break
                            if bd_res[i, min(bd_res.shape[1] - 1, j + k)] == 1 and ref_bd_i[min(bd_res.shape[1] - 1, j + k)] != 1:
                                bd_res[i, min(bd_res.shape[1] - 1, j + k)] = 0
                                break
                        bd_res[i, j] = 1
            # final check
            assert torch.sum(bd_res[i, ref_bd_i_js]) == len(ref_bd_i_js), \
                f"{torch.sum(bd_res[i, ref_bd_i_js])} {len(ref_bd_i_js)}"

    bd_res = bd_res.to(device)

    # force valid begin and end
    bd_res[:, 0] = 0
    if non_padding is not None:
        for i in range(bd_res.shape[0]):
            bd_res[i, sum(non_padding[i]) - 1:] = 0
    else:
        bd_res[:, -1] = 0

    return bd_res

class BackboneNet(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hidden_size = hidden_size = hparams['hidden_size']
        self.dropout = hparams.get('dropout', 0.0)
        updown_rates = [2, 2, 2]
        channel_multiples = [1, 1, 1]
        if hparams.get('updown_rates', None) is not None:
            updown_rates = [int(i) for i in hparams.get('updown_rates', None).split('-')]
        if hparams.get('channel_multiples', None) is not None:
            channel_multiples = [float(i) for i in hparams.get('channel_multiples', None).split('-')]
        assert len(updown_rates) == len(channel_multiples)
        # convs
        if hparams.get('bkb_net', 'conv') == 'conv':
            self.net = Unet(hidden_size, down_layers=len(updown_rates), mid_layers=hparams.get('bkb_layers', 12),
                            up_layers=len(updown_rates), kernel_size=3, updown_rates=updown_rates,
                            channel_multiples=channel_multiples, dropout=0, is_BTC=True,
                            constant_channels=False, mid_net=None, use_skip_layer=hparams.get('unet_skip_layer', False))
        # conformer
        elif hparams.get('bkb_net', 'conv') == 'conformer':
            mid_net = ConformerLayers(
                hidden_size, num_layers=hparams.get('bkb_layers', 12), kernel_size=hparams.get('conformer_kernel', 9),
                dropout=self.dropout, num_heads=4)
            self.net = Unet(hidden_size, down_layers=len(updown_rates), up_layers=len(updown_rates), kernel_size=3,
                            updown_rates=updown_rates, channel_multiples=channel_multiples, dropout=0,
                            is_BTC=True, constant_channels=False, mid_net=mid_net,
                            use_skip_layer=hparams.get('unet_skip_layer', False))

    def forward(self, x):
        return self.net(x)

class PitchDecoder(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hidden_size = hidden_size = hparams['hidden_size']
        self.dropout = hparams.get('dropout', 0.0)
        self.note_bd_out = nn.Linear(hidden_size, 1)
        self.note_bd_temperature = max(1e-7, hparams.get('note_bd_temperature', 1.0))

        # note prediction
        self.pitch_attn_num_head = hparams.get('pitch_attn_num_head', 1)
        self.multihead_dot_attn = nn.Linear(hidden_size, self.pitch_attn_num_head)
        self.post = ConvBlocks(hidden_size, out_dims=hidden_size, dilations=None, kernel_size=3,
                               layers_in_block=1, c_multiple=1, dropout=self.dropout, num_layers=1,
                               post_net_kernel=3, act_type='leakyrelu')
        self.pitch_out = nn.Linear(hidden_size, hparams.get('note_num', 100) + 4)
        self.note_num = hparams.get('note_num', 100)
        self.note_start = hparams.get('note_start', 30)
        self.pitch_temperature = max(1e-7, hparams.get('note_pitch_temperature', 1.0))

    def forward(self, feat, note_bd, train=True):
        bsz, T, _ = feat.shape

        attn = torch.sigmoid(self.multihead_dot_attn(feat))  # [B, T, C] -> [B, T, num_head]
        attn = F.dropout(attn, self.dropout, train)
        attn_feat = feat.unsqueeze(3) * attn.unsqueeze(2)  # [B, T, C, 1] x [B, T, 1, num_head] -> [B, T, C, num_head]
        attn_feat = torch.mean(attn_feat, dim=-1)  # [B, T, C, num_head] -> [B, T, C]
        mel2note = torch.cumsum(note_bd, 1)
        note_length = torch.max(torch.sum(note_bd, dim=1)).item() + 1  # max length
        note_lengths = torch.sum(note_bd, dim=1) + 1  # [B]
        # print('note_length', note_length)

        attn = torch.mean(attn, dim=-1, keepdim=True)  # [B, T, num_head] -> [B, T, 1]
        denom = mel2note.new_zeros(bsz, note_length, dtype=attn.dtype).scatter_add_(
            dim=1, index=mel2note, src=attn.squeeze(-1)
        )  # [B, T] -> [B, note_length] count the note frames of each note (with padding excluded)
        frame2note = mel2note.unsqueeze(-1).repeat(1, 1, self.hidden_size)  # [B, T] -> [B, T, C], with padding included
        note_aggregate = frame2note.new_zeros(bsz, note_length, self.hidden_size, dtype=attn_feat.dtype).scatter_add_(
            dim=1, index=frame2note, src=attn_feat
        )  # [B, T, C] -> [B, note_length, C]
        note_aggregate = note_aggregate / (denom.unsqueeze(-1) + 1e-5)
        note_aggregate = F.dropout(note_aggregate, self.dropout, train)
        note_logits = self.post(note_aggregate)
        note_logits = self.pitch_out(note_logits) / self.pitch_temperature
        # note_logits = torch.clamp(note_logits, min=-16., max=16.)     # don't know need it or not

        note_pred = torch.softmax(note_logits, dim=-1)  # [B, note_length, note_num]
        note_pred = torch.argmax(note_pred, dim=-1)  # [B, note_length]
        # for some reason, note idx maybe 130 (why?)
        note_pred[note_pred > self.note_num] = 0
        note_pred[note_pred < self.note_start] = 0

        return note_lengths, note_logits, note_pred

class MidiExtractor(nn.Module):
    def __init__(self, hparams):
        super(MidiExtractor, self).__init__()
        self.hparams = deepcopy(hparams)
        self.hidden_size = hidden_size = hparams['hidden_size']
        self.dropout = hparams.get('dropout', 0.0)
        self.note_bd_threshold = hparams.get('note_bd_threshold', 0.5)
        self.note_bd_min_gap = round(hparams.get('note_bd_min_gap', 100) * hparams['audio_sample_rate'] / 1000 / hparams['hop_size'])
        self.note_bd_ref_min_gap = round(hparams.get('note_bd_ref_min_gap', 50) * hparams['audio_sample_rate'] / 1000 / hparams['hop_size'])

        self.mel_proj = nn.Conv1d(hparams['use_mel_bins'], hidden_size, kernel_size=3, padding=1)
        self.mel_encoder = ConvBlocks(hidden_size, out_dims=hidden_size, dilations=None, kernel_size=3,
                                      layers_in_block=2, c_multiple=1, dropout=self.dropout, num_layers=1,
                                      post_net_kernel=3, act_type='leakyrelu')
        self.use_pitch = hparams.get('use_pitch_embed', True)
        if self.use_pitch:
            self.pitch_embed = Embedding(300, hidden_size, 0, 'kaiming')
            self.uv_embed = Embedding(3, hidden_size, 0, 'kaiming')
        self.use_wbd = hparams.get('use_wbd', True)
        if self.use_wbd:
            self.word_bd_embed = Embedding(3, hidden_size, 0, 'kaiming')
        self.cond_encoder = ConvBlocks(hidden_size, out_dims=hidden_size, dilations=None, kernel_size=3,
                                       layers_in_block=1, c_multiple=1, dropout=self.dropout, num_layers=1,
                                       post_net_kernel=3, act_type='leakyrelu')

        # backbone
        self.net = BackboneNet(hparams)

        # note bd prediction
        self.note_bd_out = nn.Linear(hidden_size, 1)
        self.note_bd_temperature = max(1e-7, hparams.get('note_bd_temperature', 1.0))

        # note prediction
        self.pitch_decoder = PitchDecoder(hparams)

        self.reset_parameters()

    def run_encoder(self, mel=None, word_bd=None, pitch=None, uv=None, non_padding=None):
        mel_embed = self.mel_proj(mel.transpose(1, 2)).transpose(1, 2)
        mel_embed = self.mel_encoder(mel_embed)
        pitch_embed = word_bd_embed = 0
        if self.use_pitch and pitch is not None and uv is not None:
            pitch_embed = self.pitch_embed(pitch) + self.uv_embed(uv)  # [B, T, C]
        if self.use_wbd and word_bd is not None:
            word_bd_embed = self.word_bd_embed(word_bd)
        feat = self.cond_encoder(mel_embed + pitch_embed + word_bd_embed)

        return feat

    def forward(self, mel=None, word_bd=None, note_bd=None, pitch=None, uv=None, non_padding=None, train=True):
        ret = {}
        bsz, T, _ = mel.shape

        feat = self.run_encoder(mel, word_bd, pitch, uv, non_padding)
        feat = self.net(feat)   # [B, T, C]

        # note bd prediction
        note_bd_logits = self.note_bd_out(F.dropout(feat, self.dropout, train)).squeeze(-1) / self.note_bd_temperature
        note_bd_logits = torch.clamp(note_bd_logits, min=-16., max=16.)
        ret['note_bd_logits'] = note_bd_logits  # [B, T]
        if note_bd is None or not train:
            note_bd = regulate_boundary(note_bd_logits, self.note_bd_threshold, self.note_bd_min_gap,
                                        word_bd, self.note_bd_ref_min_gap, non_padding)
            ret['note_bd_pred'] = note_bd   # [B, T]

        # note pitch prediction
        note_lengths, note_logits, note_pred = self.pitch_decoder(feat, note_bd, train)
        ret['note_lengths'], ret['note_logits'], ret['note_pred'] = note_lengths, note_logits, note_pred

        return ret

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.pitch_decoder.multihead_dot_attn.weight, mode='fan_in')
        nn.init.kaiming_normal_(self.note_bd_out.weight, mode='fan_in')
        nn.init.kaiming_normal_(self.pitch_decoder.pitch_out.weight, mode='fan_in')
        nn.init.kaiming_normal_(self.mel_proj.weight, mode='fan_in')
        nn.init.constant_(self.pitch_decoder.multihead_dot_attn.bias, 0.0)
        nn.init.constant_(self.note_bd_out.bias, 0.0)
        nn.init.constant_(self.pitch_decoder.pitch_out.bias, 0.0)


class WordbdExtractor(MidiExtractor):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.use_wbd = False
        self.word_bd_embed = None
        self.note_bd_out = self.note_bd_temperature = self.pitch_decoder = None

        self.word_bd_threshold = hparams.get('word_bd_threshold', 0.5)
        self.word_bd_min_gap = round(
            hparams.get('word_bd_min_gap', 100) * hparams['audio_sample_rate'] / 1000 / hparams['hop_size'])

        self.word_bd_out = nn.Linear(self.hidden_size, 1)
        self.word_bd_temperature = max(1e-7, hparams.get('word_bd_temperature', 1.0))
        nn.init.kaiming_normal_(self.word_bd_out.weight, mode='fan_in')
        nn.init.constant_(self.word_bd_out.bias, 0.0)

    def forward(self, mel=None, pitch=None, uv=None, non_padding=None, train=True):
        # gpu_tracker.track()
        ret = {}
        bsz, T, _ = mel.shape

        feat = self.run_encoder(mel=mel, pitch=pitch, uv=uv, non_padding=non_padding)
        feat = self.net(feat)  # [B, T, C]

        word_bd_logits = self.word_bd_out(F.dropout(feat, self.dropout, train)).squeeze(-1) / self.word_bd_temperature
        word_bd_logits = torch.clamp(word_bd_logits, min=-16., max=16.)
        ret['word_bd_logits'] = word_bd_logits  # [B, T]

        if not train:
            word_bd = regulate_boundary(word_bd_logits, self.word_bd_threshold, self.word_bd_min_gap,
                                        non_padding=non_padding)
            ret['word_bd_pred'] = word_bd   # [B, T]

        return ret

    def reset_parameters(self):
        if self.use_pitch:
            nn.init.kaiming_normal_(self.pitch_embed.weight, mode='fan_in')
            nn.init.kaiming_normal_(self.uv_embed.weight, mode='fan_in')
        nn.init.kaiming_normal_(self.mel_proj.weight, mode='fan_in')
        if self.use_pitch:
            nn.init.constant_(self.pitch_embed.weight[self.pitch_embed.padding_idx], 0.0)
            nn.init.constant_(self.uv_embed.weight[self.uv_embed.padding_idx], 0.0)


