import os
import gc
from pathlib import Path
from multiprocessing import cpu_count
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor

import torch
import torch.backends
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from hubert import HubertSoft

import audio as A


def encode_dataset(args):
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  if device == 'cuda':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

  out_path = Path('data') / args.vbank / 'units-pitdyn'
  
  print(f"Loading pretrained hubert checkpoint")
  hubert = torch.hub.load("bshall/hubert:main", "hubert_soft").to(device)

  print(f"Loading local trained hubert-pit checkpoint")
  hubert_pit = HubertSoft().to(device)
  ckpt = torch.load(f'out/{args.encode}/hubert-pit/model-best.pt', map_location=device)
  hubert_pit.load_state_dict(ckpt["hubert"]) ; del ckpt

  print(f"Loading local trained hubert-dyn checkpoint")
  hubert_dyn = HubertSoft().to(device)
  ckpt = torch.load(f'out/{args.encode}/hubert-dyn/model-best.pt', map_location=device)
  hubert_dyn.load_state_dict(ckpt["hubert"]) ; del ckpt
  
  print(f"Encoding dataset at {args.wav_path}")

  hubert.eval()
  hubert_pit.eval()
  hubert_dyn.eval()
  with torch.inference_mode(), torch.no_grad():
    for wav_fp in tqdm(list(args.wav_path.rglob("*.wav"))):
      # keep compatible with dataloader
      wav = A.load_wav(wav_fp)
      wav = A.trim_silence(wav)
      wav = A.align_wav(wav)
      wav = torch.from_numpy(wav).unsqueeze(0).unsqueeze(0)   # [B=1, C=1, T]
      wav = wav.to(device)

      units = hubert.units(wav)             # [1, T', D=256]
      embd_pit = hubert_pit.units(wav)      # [1, T', D=256]
      embd_dyn = hubert_dyn.units(wav)      # [1, T', D=256]

      units_pitdyn = torch.cat([units, embd_pit, embd_dyn], axis=-1)  # [1, T', 3*D=768]

      fp = out_path / wav_fp.relative_to(args.wav_path)
      fp.parent.mkdir(parents=True, exist_ok=True)
      np.save(fp.with_suffix(".npy"), units_pitdyn.squeeze().cpu().numpy())


def process_mel_wav(wav_fp:Path, base_dp:Path):
  wav = A.load_wav(wav_fp)
  wav = A.trim_silence(wav)
  wav = A.align_wav(wav)

  mel = A.get_mel(wav[:-1])   # [M, T]
  np.save(base_dp / 'mels' / wav_fp.with_suffix(".npy").stem, mel)

  return mel.shape[-1]

def preprocess_mel_dataset(args):
  out_path = Path('data') / args.vbank
  (out_path / 'mels').mkdir(parents=True, exist_ok=True)

  futures = []
  executor = ProcessPoolExecutor(max_workers=args.n_workers)
  print(f"Extracting features for {args.wav_path}")
  for wav_fp in args.wav_path.rglob("*.wav"):
    futures.append(executor.submit(process_mel_wav, wav_fp, out_path))
  results = [future.result() for future in tqdm(futures)]

  lengths = [length for length in results]
  frames = sum(lengths)
  hours = frames * (160 / A.hp.sample_rate) / 3600
  print(f"Found {len(lengths)} utterances, {frames} frames ({hours:.2f} hours)")


def show_qt_dataset(args):
  pit_path = Path('data') / args.vbank / 'pits'
  dyn_path = Path('data') / args.vbank / 'dyns'

  try:
    for fn in pit_path.iterdir():
      pit = np.load(pit_path / fn.name) ; pit_qt = A.qt_pit(pit) ; pit_un_qt = A.un_qt_pit(pit_qt)
      dyn = np.load(dyn_path / fn.name) ; dyn_qt = A.qt_dyn(dyn) ; dyn_un_qt = A.un_qt_dyn(dyn_qt)

      print(f'| pit - pit_un_qt | = {np.abs(pit - pit_un_qt).mean()}')
      print(f'| dyn - dyn_un_qt | = {np.abs(dyn - dyn_un_qt).mean()}')

      plt.subplot(3, 2, 1) ; plt.plot(pit)
      plt.subplot(3, 2, 2) ; plt.plot(dyn)
      plt.subplot(3, 2, 3) ; plt.plot(pit_qt)
      plt.subplot(3, 2, 4) ; plt.plot(dyn_qt)
      plt.subplot(3, 2, 5) ; plt.plot(pit_un_qt)
      plt.subplot(3, 2, 6) ; plt.plot(dyn_un_qt)
      plt.show()
  except:
    pass


def process_pitdyn_wav(wav_fp:Path, base_dp:Path):
  wav = A.load_wav(wav_fp)
  wav = A.trim_silence(wav)
  wav = A.align_wav(wav)

  pit = A.get_pit(wav[:-1])   # [T]
  dyn = A.get_dyn(wav[:-1])   # [T]
  assert len(pit) == len(dyn)

  np.save(base_dp / 'pits' / wav_fp.with_suffix(".npy").stem, pit)
  np.save(base_dp / 'dyns' / wav_fp.with_suffix(".npy").stem, dyn)

  return len(pit), (pit.max(), pit.min(), dyn.max(), dyn.min())

def preprocess_pitdyn_dataset(args):
  out_path = Path('data') / args.vbank
  (out_path / 'pits').mkdir(parents=True, exist_ok=True)
  (out_path / 'dyns').mkdir(parents=True, exist_ok=True)

  futures = []
  executor = ProcessPoolExecutor(max_workers=args.n_workers)
  print(f"Extracting features for {args.wav_path}")
  for wav_fp in args.wav_path.rglob("*.wav"):
    futures.append(executor.submit(process_pitdyn_wav, wav_fp, out_path))
  results = [future.result() for future in tqdm(futures)]

  lengths = [length for length, _ in results]
  frames = sum(lengths)
  hours = frames * (160 / A.hp.sample_rate) / 3600
  print(f"Found {len(lengths)} utterances, {frames} frames ({hours:.2f} hours)")

  with open(out_path / 'stats.txt', 'w', encoding='utf-8') as fh:
    fh.write(f'n_frames: {frames}\n')
    fh.write(f'n_hours: {hours}\n')
    fh.write(f'pit_max: {max([stats[0] for _, stats in results])}\n')
    fh.write(f'pit_min: {min([stats[1] for _, stats in results])}\n')
    fh.write(f'dyn_max: {max([stats[2] for _, stats in results])}\n')
    fh.write(f'dyn_min: {min([stats[3] for _, stats in results])}\n')


if __name__ == "__main__":
  VBANKS = os.listdir('data')   # where train data locates

  parser = ArgumentParser()
  parser.add_argument("vbank", metavar='vbank', choices=VBANKS, help='voice bank name')
  parser.add_argument("--pitdyn", action='store_true', help='generated acoustic features pit/dyn')
  parser.add_argument('--show_qt', action='store_true', help='show pit/dyn and its qt version, checking it ok')
  parser.add_argument("--mel", action='store_true', help='generated acoustic feature melspec')
  parser.add_argument("--encode", default='databaker', help='giving ref dataset, generated HuBERT hidden-units & HuBERT-pitdyn embeds combination')
  parser.add_argument("--n_workers", type=int, default=4)
  args = parser.parse_args()

  args.wav_path = Path('data') / args.vbank / 'wavs'

  if args.pitdyn:
    preprocess_pitdyn_dataset(args)
  elif args.show_qt:
    show_qt_dataset(args)
  elif args.mel:
    preprocess_mel_dataset(args)
  elif args.encode:
    encode_dataset(args)
  else:
    raise ValueError('either of --pitdyn, --mels, --encode, --show_qt must be set')
