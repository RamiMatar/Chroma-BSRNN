# Music Source Separation with Harmonic Awareness using Band Split RNN

This project builds a music source separation model based on the BSRNN architecture proposed by Y. Luo, et. al. (2022), which achieves state of the art results on MusDB on most SDR metrics. This project aims to improve the BSRNN architecture by using a high frequency-resolution spectrogram to calculate an octave-aware chromagram which is used to create harmonic attention information to augment the main spectrogram RNN path proposed in BSRNN. Pretrained models and paper will be posted soon.

# Primary References
1. Luo, Yi, and Jianwei Yu. "Music source separation with band-split rnn." IEEE/ACM Transactions on Audio, Speech, and Language Processing (2023).
2. Luo, Yi, Zhuo Chen, and Takuya Yoshioka. "Dual-path rnn: efficient long sequence modeling for time-domain single-channel speech separation." ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2020.


## Usage - Training and Evaluation

First, make sure to activate the conda env

```bash
conda env create -f environment.yml
conda activate mss
```
Make sure to download the model checkpoints and dataset (MusDB18HQ)

Consider using the following notebooks :
SourceSepEval.ipynb will let you test any of the three models against the musdb18hq test dataset.
SourceSepDemo.ipynb will let you test any of the models for any song you want and visualize the output spectrograms.


To train the model: 
```bash
 python trainer.py --chroma_version attention --batch_size 1
```
This is what you can specify, DDP will be used automatically if multiple GPUs are available.
- `--epochs`: Total number of epochs to train the model (default: 10).
- `--seed`: Seed for pseudorandom generation (default: 42).
- `--writer_dir`: Directory for TensorBoard logs (default: tb_logs).
- `--load_model`: Path to a checkpoint to load the model from.
- `--save_model`: Path to save checkpoints (default: checkpoint.pt).
- `--batch_size`: Input batch size on each device (default: 4).
- `--num_workers`: Number of worker processes for each dataloader (default: 0).
- `--validation_per_n_epoch`: Number of training epochs before doing a validation dataset iteration (default: 5).
- `--epoch_size`: Number of random segments to sample from the dataset for each epoch (default: 1000).
- `--chroma_version`: Chroma version to use, options are 'attention', 'fc_group', or other (default: attention).


## Evaluation

To evaluate the separated sources, you can use the `eval.py` script. Follow these steps:

1. Make sure you have the trained model checkpoint available.

2. Run the evaluation script:

```bash
python eval.py --model bsrnn --model_path path/to/model/checkpoint --song_path mixture.wav --source_path vocals.wav --offset 0.0 --length 30 --eval False --full_eval_mode False --plot_spectrograms False --force_mono True
```

- `--model`: Model type to use, options are 'bsrnn', 'sa', or 'fc' (default: bsrnn).
- `--model_path`: Path to the model checkpoint.
- `--song_path`: Path to the song to separate.
- `--source_path`: Path to the source to evaluate against.
- `--offset`: Offset in seconds to start separating from (default: 0.0).
- `--length`: Length in seconds to separate (default: 30).
- `--eval`: Whether to evaluate the separated source (default: False).
- `--full_eval_mode`: Whether to evaluate the separated source using museval (default: False).
- `--plot_spectrograms`: Whether to plot the spectrograms of the mixture, source, and estimate (default: False).
- `--force_mono`: Whether to force the input to be mono (default: True).



