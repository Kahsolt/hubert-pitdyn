#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/14 

# keep in order
import os

src_dp = 'test'
dst_dp = 'gen'
html_fn = 'index.html'
vbanks = [      # keep order
  # speech
  'ljspeech',
  'databaker',
  # vocal
  'sou',
  'len',
  'lemi',
  'hana',
  'ema',
  'urushi',
  'lansi',
  # instrumental
  'piano',
]
wav_fns = [      # keep order
  # speech
  'LJ001-0001.wav',                   # LJSpeech
  '000001.wav',                       # Databaker
  'BAC009S0724W0121.wav',             # AIShell
  'SSB00050353.wav',                  # AIShell3
  '000df995-51ab-4a1d-852c-c288e0300bf7.wav',   # PrimeWords
  'common_voice_ab_19904194.wav',     # Mozilla CommonVoice
  # vocal (recllst)
  '_べべぶぼべぼぼ.wav',                # UTAU - ema
  'D#4_しゅしゅししょしゃしょし.wav',    # UTAU - hana
  '_きゅんきゅきゅきぇきゅきょきゅ.wav',  # UTAU - urushi
  'kan_kan_kan.wav',                   # UTAU - lansi2
  # vocal (song)
  '[Hana] FLOWER_1_歌声-トラック_tune-8.wav',  # UTAU - hana
  '[Ema] さがり花_1_歌声-トラック-6.wav',       # UTAU - ema
  '[Len] テロル_vocal-4.wav',                  # Vocaloid - len
  # instrumental
  'Ⅱ 小快板-14.wav',                  # piano
  '十二音小夜曲-10.wav',                # strings 
]

if __name__ == '__main__':
  html_skel = '''<!DOCTYPE>
<html>
<head>
  <title>Test Soft-VC Voice Conversion</title>
  <style>
    table, th, td {
      border: 1px solid black;
    }
  </style>
</head>
<body>
<table>
%s
</table>
</body>
</html>
'''

  mk_tr    = lambda text: f'<tr>\n{text}\n</tr>'
  mk_td    = lambda text: f'  <td>{text}</td>'
  mk_audio = lambda src:  f'<audio src="{src}" controls></audio>'
  mk_p     = lambda text: f'<p>{text}</p>'
  mk_div   = lambda text: f'<div>{text}</div>'
  mk_span  = lambda text: f'<span>{text}</span>'
  
  mk_card = lambda fp, title: mk_div(f'{mk_audio(fp)}{mk_p(title)}')

  trs = []
  for fn in wav_fns:        # add rows
    tds = []

    name, ext = os.path.splitext(fn)

    # original
    fp = f'{src_dp}/{fn}'
    if os.path.exists(fp):
      tds.append(mk_td(mk_card(fp, f'{name} => original')))
    else:
      tds.append(mk_td(mk_p(f'missing {fp}')))

    # converted
    for vbank in vbanks:    # add cols
      fp = f'{dst_dp}/{name}_{vbank}{ext}'
      if os.path.exists(fp):
        tds.append(mk_td(mk_card(fp, f'{name} => {vbank}')))
      else:
        tds.append(mk_td(mk_p(f'missing {fp}')))

    trs.append(mk_tr('\n'.join(tds)))

  html_table = '\n'.join(trs)
  html = html_skel % html_table

  with open(html_fn, 'w', encoding='utf-8') as fh:
    fh.write(html)
