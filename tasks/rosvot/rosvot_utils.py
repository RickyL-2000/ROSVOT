import numpy as np

def regulate_real_note_itv(note_itv, note_bd, word_bd, word_durs, hop_size, audio_sample_rate):
    # regulate note_itv in seconds according to the correspondence between note_bd and word_bd
    assert note_itv.shape[0] == np.sum(note_bd) + 1
    assert np.sum(word_bd) <= np.sum(note_bd)
    assert word_durs.shape[0] == np.sum(word_bd) + 1, f"{word_durs.shape[0]} {np.sum(word_bd) + 1}"
    word_bd = np.cumsum(word_bd) * word_bd  # [0,1,0,0,1,0,0,0] -> [0,1,0,0,2,0,0,0]
    word_itv = np.zeros((word_durs.shape[0], 2))
    word_offsets = np.cumsum(word_durs)
    note2words = np.zeros(note_itv.shape[0], dtype=int)
    for idx in range(len(word_offsets) - 1):
        word_itv[idx, 1] = word_itv[idx + 1, 0] = word_offsets[idx]
    word_itv[-1, 1] = word_offsets[-1]
    note_itv_secs = note_itv * hop_size / audio_sample_rate
    for idx, itv in enumerate(note_itv):
        start_idx, end_idx = itv
        if word_bd[start_idx] > 0:
            word_dur_idx = word_bd[start_idx]
            note_itv_secs[idx, 0] = word_itv[word_dur_idx, 0]
            note2words[idx] = word_dur_idx
        if word_bd[end_idx] > 0:
            word_dur_idx = word_bd[end_idx] - 1
            note_itv_secs[idx, 1] = word_itv[word_dur_idx, 1]
            note2words[idx] = word_dur_idx
    note2words += 1  # mel2ph fashion: start from 1
    return note_itv_secs, note2words

def regulate_ill_slur(notes, note_itv, note2words):
    res_note2words = []
    res_note_itv = []
    res_notes = []
    note_idx = 0
    note_idx_end = 0
    while True:
        if note_idx > len(notes) - 1:
            break
        while note_idx <= note_idx_end < len(notes) and note2words[note_idx] == note2words[note_idx_end]:
            note_idx_end += 1
        res_note2words.append(note2words[note_idx])
        res_note_itv.append(note_itv[note_idx].tolist())
        res_notes.append(notes[note_idx])
        for idx in range(note_idx+1, note_idx_end):
            if notes[idx] == notes[idx-1]:
                res_note_itv[-1][1] = note_itv[idx][1]
            else:
                res_note_itv.append(note_itv[idx].tolist())
                res_note2words.append(note2words[idx])
                res_notes.append(notes[idx])
        note_idx = note_idx_end
    res_notes = np.array(res_notes, dtype=notes.dtype)
    res_note_itv = np.array(res_note_itv, dtype=note_itv.dtype)
    res_note2words = np.array(res_note2words, dtype=note2words.dtype)
    return res_notes, res_note_itv, res_note2words

def bd_to_idxs(bd):
    # bd [T]
    idxs = []
    for idx in range(len(bd)):
        if bd[idx] == 1:
            idxs.append(idx)
    return idxs

def bd_to_durs(bd):
    # bd [T]
    last_idx = 0
    durs = []
    for idx in range(len(bd)):
        if bd[idx] == 1:
            durs.append(idx - last_idx)
            last_idx = idx
    durs.append(len(bd) - last_idx)
    return durs

