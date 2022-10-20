# hubert-pitdyn

    Try using HuBERT to encode pitch and dynamic information, towards pitdyn-guided Soft-VC.

----

⚠ 发现这似乎并没有什么卵用 😓，放弃了（  
⚠ 不知道是不是我训练 recipe 没调好，hubert-pitdyn 的准确率只有 50% 多一点点，而且修改之后的声学模型还是接近从 0 开始训练的  
⚠ 这使得合成的音高错误和因音素错误都比原始的 hubert 多……  
⚠ 感觉还是得从更加庞大的预训练的升学模型中去微调 🤔，以重训练hypernetwork或者embedding的方式 :(  

In the original Soft-VC architecture, **HuBERT** is used for obtaining speech content while discarding vocal timbre.  It applies k-Means to cluster similar melspec frames all to one output, achieving these two goals above.  However, the melspec similarity is measured by Euclidian distance on spectrogram,  so that the same vowel in linguistic might be far enough under Euclidian distance, thus been divided in to different clusters.  In this case, **train dataset variety is crucial** to train a satisfactory model.  

Things get tough when you wants to model some timbre, but only has time-limited even pitch/dynamic-non-variant data (e.g. UATU voice bank recordings).  The original Soft-VC cannot handle this, synthesizing audio totally flat in pitch (robot-like).  Hence we try to introduce acoustic features (i.e. **pitch** and **dynamic**) along with HuBERT's hidden-units (which traditionally represents **content**).  

As a result, We again use HuBERT to train discrete pitch and dynamic embeddings, combining `content-units` and `acoustic-embeddings` to feed the downstream acoustic model.

### Quick Start

⚪ Train your own vbank

The idea:

  - use standard dataset `DataBaker` to train hubert-pitdyn, note that a pitch/dynamic-variant dataset is necessary to learn the acoustic aprior. 
  - use voice bank recordings `UATU - はなinit` to train acoustic-model, which is a pitch/dynamic-non-variant dataset. 
  - chain them up with pretrained HiFiGAN, now it's possible transfer any audio to timbre of  `はなinit` with proper pitch & dynamic response.

#### train hubert-pitdyn

NOTE: you can use my pretrained weights on `DataBaker`, download from [here]() and put under `out\databaker` folder

- download dataset `DataBaker` or `LJSpeech`
- train hubert-pitdyn on a large dataset (e.g.: databaker)
  - preprocess with `preprocess_hubert-pitdyn.cmd <dataset> <path\to\wavs>`
    - manually modify `class hp` in 'audio.py', set `st_*` fields according to the generated file 'data\<dataset>\stats.txt' (`f0` means pit, `c0` means dyn)
  - train pit model with `python train_hubert.py --embds_dn pits <preprocessed_path> <log_path>`
  - train dyn model with `python train_hubert.py --embds_dn dyns <preprocessed_path> <log_path>`

#### train acoustic model

NOTE: this routine is nearly the same as in [soft-vc-acoustic-models](https://github.com/Kahsolt/soft-vc-acoustic-models)

- train acoustic model on your small target timbre dataset (e.g.: hana)
  - preprocess with `preprocess_acoustic.cmd <vbank> <path\to\wavs> <dataset>`
  - train with `python train_acoustic.py <vbank> <log_path>`

Here's a full concrete example:

```powershell
# Step 1: train hubert-pitdyn
# <dataset>=databaker
# <path\to\wavs>=C:\Data\BZNSYP\Wave
preprocess_hubert-pitdyn.cmd databaker C:\Data\BZNSYP\Wave

# <preprocessed_path>=data\<dataset>
# <log_path>=out\<dataset>\hubert-[pit|dyn]
python train_hubert.py --embds_dn pits data\databaker out\databaker\hubert-pit
python train_hubert.py --embds_dn dyns data\databaker out\databaker\hubert-dyn

# Step 2: pretrain acoustic model on a larger dataset
# <vbank>=databaker
# <path\to\wavs>=C:\Data\databaker\wavs
# <dataset>=databaker
preprocess_acoustic.cmd databaker C:\Data\BZNSYP\Wave databaker

# <vbank>=databaker
# <log_path>=out\<vbank>\acoustic
python train_acoustic.py data\databaker out\databaker\acoustic

# Step 3: refine that acoustic model on your small dataset
# <vbank>=hana
# <path\to\wavs>=C:\Data\hana\wavs
# <dataset>=databaker
preprocess_acoustic.cmd hana C:\Data\hana databaker

# <vbank>=hana
# <log_path>=out\<vbank>\acoustic
python train_acoustic.py data\hana out\hana\acoustic --resume out\databaker\acoustic\model-best.pt --refine

# Step 4: now you can infer
# <vbank>=hana
# <dataset>=databaker
# => generated under 'gen' folder
python infer.py hana --dataset databaker --input test\BAC009S0724W0121.wav
```

### Project Layout

```
.
├── thesis/                   // 参考用原始论文
├── hubert_pitdyn/            // 声学编码器模型代码
├── acoustic/                 // 声学模型代码
├── data/                     // 训练用数据文件
│   ├── <vbank>/
│   │   ├── wavs/             // 指向<wavpath>的目录软连接 (由mklink产生)
│   │   ├── units/            // preprocess产生的HuBERT特征
│   │   └── mels/             // preprocess产生的Mel谱特征
│   └── ...
├── out/                      // 模型权重保存点 + 日志统计
│   ├── <vbank>/
│   │   ├── logs/             // 日志(`*.log`) + TFBoard(`events.out.tfevents.*`)
│   │   ├── model-best.pt     // 最优检查点
│   │   ├── model-<steps>.pt  // 中间检查点
│   └── ...
├── preprocess.py             // 数据预处理代码
├── train.py                  // 训练代码
├── infer.py                  // 合成代码 (Commandline API)
├── demo.ipynb                // 编程API示例 (Programmatic API)
|── ...
├── make_train.cmd            // 预处理脚本 (仅预处理，步骤1~3)
├── make_preprocess.cmd       // 训练脚本 (仅训练，步骤4)
|── ...
├── test/                     // demo源数据集
├── gen/                      // demo生成数据集 (demo源数据集在demo声库上产生的转换结果)
├── index.html                // demo列表页面
├── make_index.py             // demo页面生成脚本 (产生index.html)
└── make_infer_test.cmd       // demo生成数据集生成脚本 (产生gen/)
```

ℹ️ These developed scripts and tools are targeted mainly for **Windows** platform, if you work on Linux or Mac, you possibly need to modify on your own :(

### References

Great thanks to the founding authors of Soft-VC! :lollipop:

```
@inproceedings{
  soft-vc-2022,
  author={van Niekerk, Benjamin and Carbonneau, Marc-André and Zaïdi, Julian and Baas, Matthew and Seuté, Hugo and Kamper, Herman},
  booktitle={ICASSP}, 
  title={A Comparison of Discrete and Soft Speech Units for Improved Voice Conversion}, 
  year={2022}
}
```

- soft-vc paper: [https://ieeexplore.ieee.org/abstract/document/9746484](https://ieeexplore.ieee.org/abstract/document/9746484)
- hubert: [https://github.com/bshall/hubert](https://github.com/bshall/hubert)

----

by Armit
2022/09/17 
