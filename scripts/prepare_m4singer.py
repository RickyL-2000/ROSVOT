# %%
import glob
import os
import json
from collections import defaultdict
from pathlib import Path
import argparse

from tqdm import tqdm
import pretty_midi

from utils.text.textgrid import TextGrid
from utils.os_utils import safe_path

ALL_PHONE = ['a', 'ai', 'an', 'ang', 'ao', 'b', 'c', 'ch', 'd', 'e', 'ei', 'en', 'eng', 'er', 'f', 'g', 'h', 'i', 'ia', 'ian', 'iang', 'iao', 'ie', 'in', 'ing', 'iong', 'iou', 'j', 'k', 'l', 'm', 'n', 'o', 'ong', 'ou', 'p', 'q', 'r', 's', 'sh', 't', 'u', 'ua', 'uai', 'uan', 'uang', 'uei', 'uen', 'uo', 'v', 'van', 've', 'vn', 'x', 'z', 'zh']
ALL_SHENGMU = ['b', 'c', 'ch', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 'sh', 't', 'x', 'z', 'zh']
ALL_YUNMU = ['a', 'ai', 'an', 'ang', 'ao',  'e', 'ei', 'en', 'eng', 'er',  'i', 'ia', 'ian', 'iang', 'iao',
             'ie', 'in', 'ing', 'iong', 'iou', 'o', 'ong', 'ou', 'u', 'ua', 'uai', 'uan', 'uang', 'uei',
             'uen', 'uo', 'v', 'van', 've', 'vn']

verbose = False

# %%
def meta_data(data_dir):
    wav_fns = sorted(
        glob.glob(f'{data_dir}/*/*.wav')
    )
    for wav_fn in wav_fns:
        sen_id = os.path.basename(wav_fn)[:-4]
        if sen_id[-4:] == '_16k':
            continue
        singer, song_name = Path(wav_fn).parent.name.split("#")
        item_name = "#".join([singer, song_name, sen_id])
        item_basename = singer + '#' + song_name
        tg_fn = f'{data_dir}/{item_basename}/{sen_id}.TextGrid'
        midi_fn = f'{data_dir}/{item_basename}/{sen_id}.mid'
        yield item_name, wav_fn, singer, tg_fn, midi_fn

def is_word(word):
    if word != '<SP>' and word != '<AP>':
        return True
    else:
        return False

def word_count(word_list):
    count = 0
    for word in word_list:
        if word != '<SP>' and word != '<AP>':
            count += 1
    return count

def find_all_index(lst, value):
    all_index = []
    for i, it in enumerate(lst):
        if it == value:
            all_index.append(i)
    return all_index

def merge_small_silence(item_name, tg_align_word, tg_align_ph):
    tg_align_word_res = []
    tg_align_ph_res = []

    ph_idx = 0
    for word_idx in range(len(tg_align_word)):
        word_ = None
        ph_ = []
        while ph_idx < len(tg_align_ph) and tg_align_word[word_idx]['xmax'] >= tg_align_ph[ph_idx]['xmax']:
            if tg_align_word[word_idx]['xmax'] - tg_align_word[word_idx]['xmin'] == tg_align_ph[ph_idx]['xmax'] - tg_align_ph[ph_idx]['xmin'] < 0.02:
                if tg_align_word[word_idx]['text'] == tg_align_ph[ph_idx]['text'] == "<SP>" or tg_align_word[word_idx]['text'] == tg_align_ph[ph_idx]['text'] == "<AP>":
                    if len(tg_align_word_res) > 0:
                        tg_align_word_res[-1]['xmax'] = tg_align_word[word_idx]['xmax']
                        tg_align_ph_res[-1]['xmax'] = tg_align_ph[ph_idx]['xmax']
                    else:
                        tg_align_word[word_idx + 1]['xmin'] = tg_align_word[word_idx]['xmin']
                        tg_align_ph[ph_idx + 1]['xmin'] = tg_align_ph[ph_idx]['xmin']
                else:
                    print(f"| wrong duration match, {item_name}")
            else:
                word_ = tg_align_word[word_idx]
                ph_.append(tg_align_ph[ph_idx])
            ph_idx += 1
        if word_ is not None and len(ph_) > 0:
            tg_align_word_res.append(word_)
            tg_align_ph_res.extend(ph_)

    return tg_align_word_res, tg_align_ph_res


    word_idx = 0
    ph_idx = 0
    while word_idx < len(tg_align_word) and ph_idx < len(tg_align_ph):
        while tg_align_word[word_idx]['xmin'] > tg_align_ph[ph_idx]['xmin']:
            ph_idx += 1
        while tg_align_ph[ph_idx]['xmin'] >= tg_align_word[word_idx]['xmax']:
            word_idx += 1
        if tg_align_word[word_idx]['xmax'] - tg_align_word[word_idx]['xmin'] == tg_align_ph[word_idx]['xmax'] - tg_align_ph[word_idx]['xmin'] < 0.02:
            if tg_align_word[word_idx]['text'] == tg_align_ph[ph_idx]['text'] == "<SP>" or tg_align_word[word_idx]['text'] == tg_align_ph[ph_idx]['text'] == "<AP>":
                tg_align_word_res[-1]['xmax'] = tg_align_word[word_idx]['xmax']
                tg_align_ph_res[-1]['xmax'] = tg_align_ph[word_idx]['xmax']
            else:
                print(f"| wrong duration match, {item_name}")
        else:
            ph_idx += 1

def deal_score(tg_align_word, note_list):
    notes_pitch_list = []
    notes_dur_list = []
    types = []
    note2words = []
    note_id = 0
    last_note_end = 0.0
    last_note_dur = 0.0
    for word_id, word in enumerate(tg_align_word):
        word_text = word['text']
        word_start = word['xmin']
        word_end = word['xmax']

        while note_id < len(note_list):
            note = note_list[note_id]
            note_start = note.start
            note_end = note.end
            note_pitch = note.pitch

            if abs(last_note_end - note_start) >= 0.002 and word_text not in ['<AP>', '<SP>']:
                note_dur = note_start - last_note_end
                assert note_dur > 0
                if note_dur < 0.025:
                    if len(notes_dur_list) == 0:
                        note_start = last_note_end
                    else:
                        last_note_dur = round(last_note_dur + note_dur, 2)

            if note_start >= word_end - 0.002:
                if word_text in ['<AP>', '<SP>']:
                    if len(notes_dur_list) > 0:
                        notes_dur_list[-1] = last_note_dur
                    note_dur = word_end - word_start
                    notes_dur_list.append(round(note_dur, 2))
                    notes_pitch_list.append(0)
                    note2words.append(word_id)
                    last_note_end = word_end
                    last_note_dur = round(note_dur, 2)
                    # types.append(1)
                break
            elif note_end <= word_end + 0.002 and note_start >= word_start - 0.002:  # multiple notes correspond to this word
                if len(notes_dur_list) > 0:
                    notes_dur_list[-1] = last_note_dur
                note_dur = note_end - note_start
                notes_dur_list.append(round(note_dur, 2))
                notes_pitch_list.append(note_pitch)
                note2words.append(word_id)
                last_note_end = note_end
                last_note_dur = round(note_dur, 2)
                note_id += 1
                continue
            else:
                break

    note = note_list[-1]
    note_end = note.end
    if note_end <= tg_align_word[-1]['xmax'] - 0.02:
        note_dur = tg_align_word[-1]['xmax'] - last_note_end
        notes_dur_list.append(round(note_dur, 2))
        notes_pitch_list.append(0)
        note_end = tg_align_word[-1]['xmax']
        note2words.append(len(tg_align_word) - 1)
    elif note_end <= float(tg_align_word[-1]['xmax']):
        notes_dur_list[-1] = notes_dur_list[-1] + tg_align_word[-1]['xmax'] - note_end
        note_end = tg_align_word[-1]['xmax']
    assert abs(tg_align_word[-1]['xmax'] - note_end) <= 0.01, 'phoneme 和 note 结尾不相同'

    return notes_pitch_list, notes_dur_list, note2words

def deal_tg(tg_align_word, tg_align_ph):
    txt_list = []
    ph_list = []
    ph_dur_list = []
    wbd_list = []
    word_durs = []
    ph2words = []
    types = []
    ph_id = 0
    word_id = 0
    while word_id < len(tg_align_word):
        word = tg_align_word[word_id]
        word['xmin'] = float(word['xmin'])
        word['xmax'] = float(word['xmax'])
        if is_word(word['text']):
            txt_list.append(word['text'])
        else:
            pass
        word_dur = round(word['xmax'] - word['xmin'], 2)
        word_durs.append(word_dur)

        while ph_id < len(tg_align_ph):
            ph = tg_align_ph[ph_id]
            ph_list.append(ph['text'])
            ph['xmax'] = float(ph['xmax'])
            ph['xmin'] = float(ph['xmin'])
            ph_dur = round(ph['xmax'] - ph['xmin'], 2)
            ph_dur_list.append(ph_dur)
            wbd_list.append(1 if ph['text'] in ALL_YUNMU + ['SP', 'AP'] else 0)

            if ph['text'] == '<AP>':
                types.append(1)
            else:
                types.append(2)

            if abs(ph['xmax'] - word['xmax']) <= (1e-5):
                ph_id += 1
                wbd_list[-1] = 1
                ph2words.append(word_id)
                break
            ph_id += 1
            ph2words.append(word_id)
        word_id += 1

    txt = "".join(t for t in txt_list if t != 'SL')

    return txt, txt_list, ph_list, ph_dur_list, types, wbd_list, word_durs, ph2words

def deal_align(item_name, ph_list, ph_dur_list, ph2words, types, notes_pitch_list, notes_dur_list, note2words):
    ep_pitches = []
    ep_notedurs = []
    ep_types = []
    ep_ph_list = []
    ep_ph_dur_list = []
    ep_ph2words = []


    for i in range(max(note2words) + 1):
        ph_idxs = find_all_index(ph2words, i)
        note_idxs = find_all_index(note2words, i)
        first_ph = ph_list[ph_idxs[0]]
        flag = False
        is_none = False
        if first_ph in ALL_SHENGMU:
            ep_pitches.append(notes_pitch_list[note_idxs[0]])
            ep_notedurs.append(notes_dur_list[note_idxs[0]])
            ep_types.append(2)
            ep_ph_list.append(ph_list[ph_idxs[0]])
            ep_ph_dur_list.append(ph_dur_list[ph_idxs[0]])
            ep_ph2words.append(ph2words[ph_idxs[0]])
            if len(note_idxs) != len(ph_idxs) - 1:
                if notes_dur_list[note_idxs[0]] > ph_dur_list[ph_idxs[0]]:
                    ep_ph_list.append(ph_list[ph_idxs[1]])
                    ep_ph_dur_list.append(notes_dur_list[note_idxs[0]] - ph_dur_list[ph_idxs[0]])
                    ep_ph2words.append(ph2words[ph_idxs[1]])
                    ep_types.append(2)
                    for j in range(len(note_idxs) - (len(ph_idxs) - 1)):
                        ep_ph_list.append(ph_list[ph_idxs[1]])
                        ep_ph_dur_list.append(notes_dur_list[note_idxs[j + 1]])
                        ep_ph2words.append(ph2words[ph_idxs[1]])
                        ep_types.append(3)
                elif notes_dur_list[note_idxs[0]] == ph_dur_list[ph_idxs[0]]:
                    flag = True
                    for j in range(len(note_idxs) - (len(ph_idxs) - 1)):
                        ep_ph_list.append(ph_list[ph_idxs[1]])
                        ep_ph_dur_list.append(notes_dur_list[note_idxs[j + 1]])
                        ep_ph2words.append(ph2words[ph_idxs[1]])
                        ep_types.append(3)
                else:
                    flag = True
                    ep_ph_dur_list[-1] = notes_dur_list[note_idxs[0]]
                    ep_pitches.append(notes_pitch_list[note_idxs[1]])
                    ep_notedurs.append(notes_dur_list[note_idxs[1]])
                    ep_types.append(3)
                    ep_ph_list.append(ph_list[ph_idxs[0]])
                    ep_ph_dur_list.append(ph_dur_list[ph_idxs[0]] - notes_dur_list[note_idxs[0]])
                    ep_ph2words.append(ph2words[ph_idxs[0]])

                    if len(note_idxs) == 2:
                        ep_ph_list.append(ph_list[ph_idxs[1]])
                        ep_ph_dur_list.append(ph_dur_list[ph_idxs[1]])
                        ep_ph2words.append(ph2words[ph_idxs[1]])
                        ep_types.append(3)
                    elif len(note_idxs) > 2:
                        ep_ph_list.append(ph_list[ph_idxs[1]])
                        ep_ph_dur_list.append(
                            notes_dur_list[note_idxs[0]] + notes_dur_list[note_idxs[1]] - ph_dur_list[ph_idxs[0]])
                        ep_ph2words.append(ph2words[ph_idxs[1]])
                        ep_types.append(3)
                        for j in range(len(note_idxs) - len(ph_idxs)):
                            ep_ph_list.append(ph_list[ph_idxs[1]])
                            ep_ph_dur_list.append(notes_dur_list[note_idxs[j + 2]])
                            ep_ph2words.append(ph2words[ph_idxs[1]])
                            ep_types.append(3)
            else:
                ep_ph_list.append(ph_list[ph_idxs[1]])
                ep_ph_dur_list.append(ph_dur_list[ph_idxs[1]])
                ep_ph2words.append(ph2words[ph_idxs[1]])
                ep_types.append(types[ph_idxs[1]])
        elif first_ph in ['<AP>', '<SP>', 'breath', 'breathe', '_NONE']:
            is_none = True
            ep_ph_list.append(ph_list[ph_idxs[0]])
            ep_ph_dur_list.append(ph_dur_list[ph_idxs[0]])
            ep_ph2words.append(ph2words[ph_idxs[0]])
            ep_types.append(1)
            ep_pitches.append(0)
            ep_notedurs.append(ph_dur_list[ph_idxs[0]])
        else:
            for j in range(len(note_idxs)):
                ep_ph_list.append(ph_list[ph_idxs[0]])
                ep_ph_dur_list.append(notes_dur_list[note_idxs[j]])
                ep_ph2words.append(ph2words[ph_idxs[0]])
                if j == 0:
                    ep_types.append(2)
                else:
                    ep_types.append(3)
        if not is_none:
            if flag:
                ep_pitches.extend([notes_pitch_list[i] for i in note_idxs[1:]])
                ep_notedurs.extend([notes_dur_list[i] for i in note_idxs[1:]])
            else:
                ep_pitches.extend([notes_pitch_list[i] for i in note_idxs])
                ep_notedurs.extend([notes_dur_list[i] for i in note_idxs])

    return ep_pitches, ep_notedurs, ep_types, ep_ph_list, ep_ph_dur_list, ep_ph2words

def process(data_dir, tgt_meta_path):
    gen = meta_data(data_dir)
    phs_meta = defaultdict(int)
    meta_out = []
    sen_dur_list = []
    spk_map = set()

    for item_idx, item in tqdm(enumerate(gen)):
        (item_name, wav_fn, singer, tg_fn, midi_fn) = item

        with open(tg_fn, "r") as f:
            tg = f.readlines()
        tg = TextGrid(tg)
        tg = json.loads(tg.toJson())

        tg_align_word = [x for x in tg['tiers'][0]['items']]
        tg_align_ph = [x for x in tg['tiers'][1]['items']]
        word_list = [xx['text'] for xx in tg_align_word]
        for idx in range(len(tg_align_word)):
            tg_align_word[idx]['xmin'] = float(tg_align_word[idx]['xmin'])
            tg_align_word[idx]['xmax'] = float(tg_align_word[idx]['xmax'])
        for idx in range(len(tg_align_ph)):
            tg_align_ph[idx]['xmin'] = float(tg_align_ph[idx]['xmin'])
            tg_align_ph[idx]['xmax'] = float(tg_align_ph[idx]['xmax'])

        if tg_align_word[-1]['xmax'] > 20 and verbose:
            print(item_name, '| sentence too long')

        mf = pretty_midi.PrettyMIDI(midi_fn)
        instru = mf.instruments[0]
        note_list = instru.notes

        tg_align_word, tg_align_ph = merge_small_silence(item_name, tg_align_word, tg_align_ph)
        notes_pitch_list, notes_dur_list, note2words = deal_score(tg_align_word, note_list)
        txt, txt_list, ph_list, ph_dur_list, types, wbd_list, word_durs, ph2words = deal_tg(tg_align_word, tg_align_ph)
        ep_pitches, ep_notedurs, ep_types, ep_ph_list, ep_ph_dur_list, ep_ph2words = \
            deal_align(item_name, ph_list, ph_dur_list, ph2words, types, notes_pitch_list, notes_dur_list, note2words)

        notes_dur_list = [round(i, 2) for i in notes_dur_list]
        ep_ph_dur_list = [round(i, 2) for i in ep_ph_dur_list]
        ep_notedurs = [round(i, 2) for i in ep_notedurs]

        ph = ep_ph_list  # 重复过后的 ph
        ph_dur = ep_ph_dur_list
        ph2word = ep_ph2words

        try:
            assert abs(sum(ep_ph_dur_list) - sum(ph_dur_list)) <= 0.04 + 1e-5, f"mismatch"
            assert abs(sum(ep_ph_dur_list) - sum(notes_dur_list)) <= 0.04 + 1e-5, f"mismatch"
            assert abs(sum(ph_dur_list) - sum(word_durs)) <= 0.02 + 1e-5, f"mismatch"
        except AssertionError:
            if verbose:
                print(f"skip [{item_idx} : {item_name}] for mismatch duration")
            continue

        # filter invalid word duration
        invalid_wd = False
        for wd in word_durs:
            if wd < 20 / 1000:
                print(f'item {item_name} has invalid small word durations')
                invalid_wd = True
                break
        if invalid_wd:
            continue

        for ii in ph_list:
            phs_meta[ii] += 1
        sen_dur_list.append(tg_align_word[-1]['xmax'])
        spk_map.add(singer)

        res = {'item_name': item_name, 'txt_raw': txt, 'txt': word_list, 'word_durs': word_durs, 'ph': ph,
               'ph_durs': ph_dur, 'ph2words': ph2word, 'types': types, 'wbd': wbd_list,
               'pitches': notes_pitch_list, 'note_durs': notes_dur_list, 'note2words': note2words, 'wav_fn': wav_fn,
               'singer': singer, 'ep_types': ep_types,
               'ep_pitches': ep_pitches, 'ep_notedurs': ep_notedurs, 'total_dur': tg_align_word[-1]['xmax']}

        meta_out.append(res)

    # stats
    print(f"Total {item_idx-1} samples, skip {item_idx-1 - len(meta_out)} samples for mismatch, resulting {len(meta_out)} samples")
    print(f"Total length {sum(sen_dur_list):.02f} seconds, {sum(sen_dur_list) / 3600:.02f} hours")

    print(phs_meta)
    ph_l = set()
    for ph in phs_meta.keys():
        ph_l.add(ph)
    ph_l.add('_None')
    ph_l.add('_others')
    ph_l = list(ph_l)
    ph_l.sort()
    print(ph_l)

    meta_path = tgt_meta_path
    json.dump(meta_out, open(meta_path, 'w'), ensure_ascii=False, indent=2)
    spk_map = {str(spk): i for i, spk in enumerate(sorted(list(spk_map)))}
    spk_map["others"] = len(spk_map)
    json.dump(spk_map, open(Path(meta_path).with_name('spk_map.json'), 'w'), ensure_ascii=False, indent=2)
    json.dump(ph_l, open(Path(meta_path).with_name('phone_set.json'), 'w'), ensure_ascii=False, indent=2)

    return meta_out, spk_map

# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        type=str,
        help='Directory of data (M4Singer). '
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/m4/metadata.json",
        help='Output path of the generated metadata. '
    )
    parser.add_argument(
        "--verbose",
        action='store_true',
        help='Verbose. '
    )
    args = parser.parse_args()

    data_dir = args.dir
    if args.output == "":
        tgt_meta_path = safe_path(os.path.join(data_dir, 'metadata.json'))
    else:
        tgt_meta_path = safe_path(args.output)
    meta, spk_map = process(data_dir, tgt_meta_path)

