# ROSVOT: Robust Singing Voice Transcription and MIDI Extraction

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2405.09940)

This is the official PyTorch implementation of [ROSVOT (ACL'24)](https://arxiv.org/abs/2405.09940), a robust automatic singing voice transcription (AST) model that serves singing voice synthesis (SVS). We provide the original design and implementation of this method, along with the model weights. **We are still working on this project, feel free to create an issue if you find any problem.** 

## What ROSVOT can do

1. **Automatically transcribe singing voices**, i.e., turn a waveform of singing voice into a sequence of note events, in the form of MIDI. 
2. **Transcribe noisy or accompanied singing voices**. In practice, you may have to deal with noisy singing voices separated from songs or movies.  As a robust AST model, ROSVOT can work under noisy environments and even transcribe raw songs with accompaniments. 
3. **Integrated note-word alignment**. If word boundaries are given (typically generated from [MFA](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner) or other ASR/forced aligner tools), the output MIDI notes are aligned with word timestamps.
4. **Multi-lingual processing**. The checkpoint we provided was trained with Mandarin datasets. However, we find that (mostly empirically) it works pretty well in other languages. Feel free to try different languages!
5. **Bulk processing**. We provide parallel transcription with multiprocessing and batched inference. 
6. **Memory-efficient transcription**. 

> Note: This method still has plenty of rooms for improvement! Therefore, we are continuing to optimize it and will release version 2.0 in the near future. 

## Dependencies

ROSVOT is tested in **python3.9**, **torch 2.1.1**, **CUDA 11.8**. Experiences have shown us that directly sharing `requirements.txt` is prone to dependency issues, so we now provide a series of install commands ([install.sh](scripts/install.sh)) for easier installation:

```shell
conda create -n rosvot python==3.9
conda activate rosvot
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tensorflow==2.9.0 tensorflow-estimator==2.9.0 tensorboardX==2.5
pip install pyyaml matplotlib==3.5 pandas pyworld==0.2.12 librosa torchmetrics
pip install mir_eval pretty_midi pyloudnorm scikit-image textgrid g2p_en npy_append_array einops webrtcvad
export PYTHONPATH=.
```

The commands above are for training and redundant for inference. If you only use ROSVOT for inference, refer to the following commands:

```shell
conda create -n rosvot python==3.9
conda activate rosvot
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install librosa tqdm matplotlib==3.5 pyyaml pretty_midi pyworld==0.2.12
export PYTHONPATH=.
```

## Model Weights

We provide a checkpoint of necessary model weights in [here](https://drive.google.com/file/d/1JNtNT37KiLq9uFQqHk7JFs-3trxd3bRh/view?usp=sharing). Download the `checkpoints.zip` and unzip it under the project root. The zip file contains the following model weights:

| Model       | Discription   | Position |
|:-------------:|:--------:|:---:|
|   ROSVOT    | MIDI extraction model | checkpoints/rosvot |
| RWBD    | word boundary prediction model  | checkpoints/rwbd |
| RMVPE  | pitch extraction model | checkpoints/rmvpe |

Note that the provided checkpoints are trained with [M4Singer](https://github.com/M4Singer/M4Singer) only, for better reproduction. 

## Inference

We provide both single-run and batched inference, where the latter is for bulk data annotation for SVS tasks. For detailed instruction, run `python inference/rosvot.py --help` or check the [source code](inference/rosvot.py).

For single-run inference, just run: 

```shell
python inference/rosvot.py -o [path-to-result-directory] -p [path-to-the-wave-file]
```

Don't forget to replace `[path-to-result-directory]` with the output directory and `[path-to-the-wave-file]` with the target waveform file. Add `--save_plot` flag if you want visualizations. For a detailed output, add `-v`. In this mode, a word boundary predictor is automatically applied to extract potential word boundaries. 

For batched parallel extraction, you need to provide a manifest `.json` file that contains the paths of waveforms. The `.json` file should contain a `list` of `dict`s, where each `dict` has at least two attributes: `item_name` and `wav_fn`, where the former is the unique identifier of this file and the latter is the audio path. An optional attribute `word_durs` is used to provide desired word boundaries. In other words, if `word_durs` is absent, the word boundary predictor will be automatically applied. An example manifest file looks like this:

```json
[
  {
    "item_name": "Alto-1#newboy#0000",
    "wav_fn": "~/datasets/m4singer/Alto-1#newboy/0000.wav",
    "word_durs":  [0.1, 0.17, 0.65, 0.19, 0.33, 0.23, 0.14, 0.5, 0.5, 0.34, 0.49, 0.39, 0.39, 0.48, 0.1]
  }
]
```

Once the manifest is available, run the following command for bulk extraction:

```shell
CUDA_VISIBLE_DEVICES=[your-gpus] python inference/rosvot.py -o [path-to-result-directory] --metadata [path-to-manifest-file]
```

You can use flags `--bsz` and `--max_tokens` to control memory usage of each GPU, while `--ds_workers` can be used to accelerate data preprocessing on the CPU side. Note that since we use DDP for bulk processing, the initialization of the process is pretty slow (slower with more ds_workers). You can override the word boundary condition using `--apply_rwbd`. 

## Train

### Data Preparation

We provide a data preprocessing pipeline that uses [M4Singer](https://github.com/M4Singer/M4Singer) as an example. The pipeline can be transferred to arbitrary datasets as long as they contain similar annotations. Once you download and unzip the M4Singer dataset, run the preprocessing script to generate a manifest file for training:

```shell
python scripts/prepare_m4singer.py --dir [path-to-m4singer] --output data/processed/m4/metadata.json
```

Once you have the manifest file `data/processed/m4/metadata.json`, you can start binarizing the dataset. However, you may want to take a look at [configs/rosvot.yaml](configs/rosvot.yaml) first, where the `test_prefixes` attribute indicates the prefixes of `item_name`s of the desired test samples. Feel free to change it. In our setting, the valid set and test set could be the same, since the actual test set is OOD. To binarize the dataset into a single file, run:

```shell
CUDA_VISIBLE_DEVICES=[your-gpus] python data_gen/run.py --config configs/rosvot.yaml
```

Additionally, we need external noise datasets for robust training (you can disable noise injection by simply setting the `noise_prob` hyper-parameter in [configs/rosvot.yaml](configs/rosvot.yaml) to 0.0). We use [MUSAN](https://www.openslr.org/17/) dataset as the noise source. Once you download and unzip the dataset, replace the value of the `raw_data_dir` attribute in [configs/musan.yaml](configs/musan.yaml) with the current path of MUSAN, and run the following command to binarize the noise source:

```shell
python data_gen/run.py --config configs/musan.yaml
```

### Training

To train a MIDI extraction model, run:

```shell
CUDA_VISIBLE_DEVICES=[your-gpus] python tasks/run.py --config configs/rosvot.yaml --exp_name [your-exp-name] --reset
```

where the `--exp_name` is the experiment tag. The checkpoints and logs are saved in `checkpoints/[your-exp-name]/`.

To train a word boundary prediction model, run:

```shell
CUDA_VISIBLE_DEVICES=[your-gpus] python tasks/run.py --config configs/rwbd.yaml --exp_name [your-exp-name] --reset
```

## Acknowledgements

This implementation uses parts of the code from the following GitHub repos:
[NATSpeech](https://github.com/NATSpeech/NATSpeech),
[DiffSinger](https://github.com/MoonInTheRiver/DiffSinger).

## Citation

If you find this code useful in your research, please cite our work:

```bibtex
@misc{li2024robust,
      title={Robust Singing Voice Transcription Serves Synthesis}, 
      author={Ruiqi Li and Yu Zhang and Yongqi Wang and Zhiqing Hong and Rongjie Huang and Zhou Zhao},
      year={2024},
      eprint={2405.09940},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}
```

## Disclaimer ##
Any organization or individual is prohibited from using any technology mentioned in this paper to generate someone's speech/singing without his/her consent, including but not limited to government leaders, political figures, and celebrities. If you do not comply with this item, you could be in violation of copyright laws.

