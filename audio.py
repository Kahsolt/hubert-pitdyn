#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/01/07 

import os
TMP_PATH = os.path.join(os.environ['TMP'], 'librosa-cache')
if not os.path.exists(TMP_PATH): os.mkdir(TMP_PATH)
os.environ['LIBROSA_CACHE_DIR'] = TMP_PATH
os.environ['LIBROSA_CACHE_LEVEL'] = '10'
import librosa as L

import numpy as np
from scipy.io import wavfile

import seaborn as sns
import matplotlib.pyplot as plt


class hp:

  # NOTE: these are set compatible to other soft-vc models, DO NOT MODIFY!
  sample_rate  = 16000
  window_fn    = 'hann'
  n_fft        = 1024
  win_length   = 1024
  hop_length   = 160       # for mel
  hop_length2  = 320       # for pit/dyn, HuBERT's fx_net will do x320 downsample 
  n_mel        = 128
  fmin         = 0
  fmax         = 8000

  # these are modifiable
  trim_db      = 50
  f0min        = 20        # freq band for f0-detect
  f0max        = 550

  # NOTE: tune these stats for your dataset!
  st_f0max = 551.25
  st_f0min = 43.15068435668945
  st_c0max = 0.3349683880805969
  st_c0min = 0.0

  # qt bins
  n_bin_pit = int(np.ceil(L.hz_to_midi(st_f0max)) - np.floor(L.hz_to_midi(st_f0min))) + 1
  n_bin_dyn = 32

  # n_bins should <= 100 (because I didn't modify output layer of the default HuBERT)
  assert n_bin_pit <= 100
  assert n_bin_dyn <= 100


def load_wav(path):  # float values in range (-1,1)
  y, _ = L.load(path, sr=hp.sample_rate, mono=True, res_type='kaiser_best')
  return y.astype(np.float32)     # [T,]


def save_wav(wav, path):
  wavfile.write(path, hp.sample_rate, wav)


def trim_silence(wav, frame_length=512, hop_length=128):
  # 人声动态一般高达55dB
  return L.effects.trim(wav, top_db=hp.trim_db, frame_length=frame_length, hop_length=hop_length)[0]


def align_wav(wav, r=hp.hop_length2):
  d = len(wav) % r
  if d != 0:
    wav = np.pad(wav, (0, (r - d)))
  return wav


def get_pit(y):
  pit = L.yin(y, fmin=hp.f0min, fmax=hp.f0max, frame_length=hp.win_length, hop_length=hp.hop_length2)
  return pit.astype(np.float32)       # [T,]


def get_dyn(y):
  dyn = L.feature.rms(y=y, frame_length=hp.win_length, hop_length=hp.hop_length2)[0]
  return dyn.astype(np.float32)       # [T,]


def get_mel(y, clamp_low=True):
  M = L.feature.melspectrogram(y=y, sr=hp.sample_rate, n_fft=hp.n_fft, hop_length=hp.hop_length, win_length=hp.win_length, 
                               n_mels=hp.n_mel, fmin=hp.fmin, fmax=hp.fmax,
                               window=hp.window_fn, power=1, htk=False, norm='slaney')
  mel = np.log(M.clip(min=1e-5) if clamp_low else M)
  return mel.astype(np.float32)      # [M, T]


def qt_pit(pit):
  pit = np.asarray([L.hz_to_midi(f) - L.hz_to_midi(hp.st_f0min) for f in pit])
  pit = pit.clip(0, hp.n_bin_pit - 1)
  return pit.astype(np.int32)         # [T,]


def qt_dyn(dyn):
  dyn = (dyn - hp.st_c0min) / (hp.st_c0max - hp.st_c0min)
  dyn = dyn * hp.n_bin_dyn
  dyn = dyn.clip(0, hp.n_bin_dyn - 1)
  return dyn.astype(np.int32)         # [T,]


def un_qt_pit(pit_qt):
  pit = np.asarray([L.midi_to_hz(f + L.hz_to_midi(hp.st_f0min)) for f in pit_qt])
  return pit.astype(np.float32)         # [T,]


def un_qt_dyn(dyn_qt):
  dyn = dyn_qt / hp.n_bin_dyn
  dyn = dyn * (hp.st_c0max - hp.st_c0min) + - hp.st_c0min
  return dyn.astype(np.float32)         # [T,]
