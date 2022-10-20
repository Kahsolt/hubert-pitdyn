#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/12 

import os
from argparse import ArgumentParser
from traceback import print_exc

import torch

from hubert import HubertSoft
from acoustic import AcousticModel
import audio as A

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def convert(args):
  hubert     = torch.hub.load("bshall/hubert:main", "hubert_soft"         ).to(device)
  hubert_pit = HubertSoft()                                                .to(device)
  hubert_dyn = HubertSoft()                                                .to(device)
  acoustic   = AcousticModel()                                             .to(device)
  hifigan    = torch.hub.load("bshall/hifigan:main", "hifigan_hubert_soft").to(device)

  ckpt = torch.load(f'out/{args.dataset}/hubert-pit/model-best.pt', map_location=device)
  hubert_pit.load_state_dict(ckpt["hubert"]); del ckpt
  ckpt = torch.load(f'out/{args.dataset}/hubert-dyn/model-best.pt', map_location=device)
  hubert_dyn.load_state_dict(ckpt["hubert"]); del ckpt
  ckpt = torch.load(f'out/{args.vbank}/acoustic/model-best.pt', map_location=device)
  acoustic.load_state_dict(ckpt["acoustic-model"]); del ckpt

  if os.path.isfile(args.input):
    wav_fps = [args.input]
  else:
    wav_fps = [os.path.join(args.input, fn) for fn in os.listdir(args.input)]
  os.makedirs(args.out_path, exist_ok=True)
  
  with torch.inference_mode():
    for wav_fp in wav_fps:
      try:
        wav = A.load_wav(wav_fp)
        wav = A.trim_silence(wav)
        wav = A.align_wav(wav)
        wav = torch.from_numpy(wav).unsqueeze(0).unsqueeze(0)
        wav = wav.to(device)
        
        units = hubert.units(wav)
        embd_pit = hubert_pit.units(wav)
        embd_dyn = hubert_dyn.units(wav)
        units_pitdyn = torch.cat([units, embd_pit, embd_dyn], axis=-1)
        mel = acoustic.generate(units_pitdyn).transpose(1, 2)
        target = hifigan(mel)

        y_hat = target.squeeze().cpu().numpy()
        name, ext = os.path.splitext(os.path.basename(wav_fp))
        save_fp = os.path.join(args.out_path, f'{name}_{args.vbank}{ext}')
        A.save_wav(y_hat, save_fp)
        print(f'>> {save_fp}')
      except Exception as e:
        print_exc()
        #print(f'<< [Error] {e}')
        print(f'<< ignore file {wav_fp}')


if __name__ == '__main__':
  VBANKS = os.listdir('out')   # where ckpt locates

  parser = ArgumentParser()
  parser.add_argument("vbank", metavar='vbank', default='databaker', choices=VBANKS, help='voice bank name')
  parser.add_argument("--dataset", default='databaker', choices=VBANKS, help='hubert-pitdyn dataset')
  parser.add_argument("--input", default='test', help='input file or folder for conversion')
  parser.add_argument("--out_path", default='gen', help='output folder for converted wavfiles')
  args = parser.parse_args()

  convert(args)
