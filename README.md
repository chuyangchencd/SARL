# SARL — Spatial Audio Representation Learning

A **probing benchmark** for how well pretrained audio encoders capture spatial
structure. SARL freezes a backbone and trains a lightweight linear probe on each
task, measuring what spatial information is linearly decodable from its frozen
embeddings.

Seven tasks in two families (run separately):

| family | tasks | metric |
|--------|-------|--------|
| **source** | `azimuth`, `elevation`, `distance`, `event` | normalized MAE / macro-F1 |
| **room** | `rt60`, `volume`, `shape` | normalized MAE / macro-F1 |

> **[Probing Spatial Structure in Pretrained Audio Representations](https://arxiv.org/abs/2606.05544)**<br>
> Chuyang Chen, Sivan Ding, Adrian S. Roman, Juan P. Bello · Interspeech 2026

**Dataset (pre-rendered RIRs):** [huggingface.co/datasets/chuyangchenn/SARL](https://huggingface.co/datasets/chuyangchenn/SARL)

## Installation

```bash
pip install -r requirements.txt
```

Core evaluation needs `torch`, `torchaudio`, `numpy`. Building the dataset also needs
`soundfile`, `scipy`, `sofar`; downloading the RIRs needs `huggingface_hub`.

## Repository layout

```
tasks.py            task definitions: bins, metrics, and value decoding
metrics.py          scoring: normalized MAE, macro-F1, and baseline normalization
data/dataset.py     spatial scenes synthesized on the fly, plus the loader
models/
  base.py           BackboneWrapper — the interface you implement
  baselines.py      weight-free reference backbone (rawfeat)
  registry.py       register and build backbones by name
  template.py       a template to copy for a new backbone
probe/
  model.py          frozen backbone + linear/MLP head, and the naive predictor
  run.py            train and evaluate entry point
preprocessing/      build audio/ and ambient/ from public datasets
```

## Data setup

Assemble `data_root/` from four parts (on-disk format in [Data format](#data-format)):

```
data_root/
├── audio/        built from ESC-50 / MUSAN / UrbanSound8K
├── ambient/      built from TAU-SNoise
├── rir_source/   downloaded (source-task RIRs)
└── rir_room/     downloaded (room-task RIRs)
```

**1. Source clips** — download [ESC-50](https://github.com/karolpiczak/ESC-50),
[MUSAN](https://www.openslr.org/17/), and
[UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html), then build
the 7-class pool (mono, 24 kHz, 10 s, −24 dBFS):

```bash
python -m preprocessing.build esc50        --source /path/to/ESC-50       --out data_root
python -m preprocessing.build musan        --source /path/to/musan        --out data_root
python -m preprocessing.build urbansound8k --source /path/to/UrbanSound8K --out data_root
```

**2. Ambient noise** — download [TAU-SNoise](https://zenodo.org/records/6408611) and
build the `foa`/`mic`/`binaural` ambient. The binaural ambient is decoded from FOA with
an HRTF; we use `FABIAN_HRIR_measured_HATO_0.sofa` from the
[FABIAN HRTF database](https://depositonce.tu-berlin.de/items/bff6568a-5735-4ebc-b3fa-ac10707b7beb):

```bash
python -m preprocessing.preprocess_ambient --source /path/to/TAU-SNoise_DB \
    --out data_root --hrtf /path/to/FABIAN_HRIR_measured_HATO_0.sofa
```

**3. RIRs** — download the pre-rendered RIRs from the
[SARL dataset on HuggingFace](https://huggingface.co/datasets/chuyangchenn/SARL)
(they're stochastic to regenerate, so everyone uses the same released set). They ship
as per-format tar archives; unpack them in place:

```bash
huggingface-cli download chuyangchenn/SARL --repo-type dataset --local-dir data_root
cd data_root && for t in rir_*.tar; do tar xf "$t"; done && rm rir_*.tar
```

To fetch only the format you need, add `--include "*_foa.tar" "*/metadata.json"` (etc.).

Preprocessing is deterministic given `--seed`, so the same sources reproduce the same
pools as ours.

## Quick start

Evaluate a weight-free baseline end to end (trains the heads, writes a JSON of scores):

```bash
python -m probe.run --backbone rawfeat_logmel_binaural \
    --tasks azimuth elevation distance event --data_root data_root
```

The naive chance baseline (no training, the normalization reference):

```bash
python -m probe.run --backbone naive_random --tasks rt60 volume shape \
    --data_root data_root --audio_format foa --sample_rate 24000
```

Re-evaluate saved heads without retraining:

```bash
python -m probe.run --backbone rawfeat_logmel_binaural \
    --tasks azimuth elevation distance event --data_root data_root --eval_only
```

Defaults match the paper (20 epochs, batch 32, Adam-W lr 1e-4, cosine, Gaussian soft
labels for continuous tasks). Source and room tasks must be run separately.

## Adding your own backbone

Implement one method. Copy `models/template.py` and fill it in:

```python
from models.base import BackboneWrapper
from models.registry import register

class MyBackbone(BackboneWrapper):
    def __init__(self):
        super().__init__()
        self.sample_rate = 24000       # rate your encoder wants
        self.audio_format = "binaural" # "stereo" | "binaural" | "foa"
        self.output_dim = 768          # embedding dim D
        self.encoder = load_my_encoder(...)

    def forward_features(self, audio):     # audio [B, C, T]
        return self.encoder(audio)         # -> [B, T, D] or [B, D]

@register("my_encoder")
def _build():
    return MyBackbone()
```

The probe freezes the backbone, mean-pools the features (override `aggregate` to
change that), normalizes, and trains the head — you only provide features. Make sure
your module is imported before you run (add it to `models/__init__.py`), then:

```bash
python -m probe.run --backbone my_encoder --tasks azimuth elevation distance event \
    --data_root data_root
```

## Checkpoints

Training saves **only the trained heads** (not the frozen backbone) — one small file
per run, each head at its own best-validation epoch. `--eval_only` reloads them onto
the freshly built backbone.

## Data format

All data lives under `data_root`, split into `train`/`val`/`test`:

```
audio/<split>/<class>/*.wav       mono source clips (7 event classes)
rir_source/<split>/<fmt>/*.npy    source RIRs [C,N,T] + <split>/metadata.json
rir_room/<split>/<fmt>/*.wav      room RIRs   [C,T]   + <split>/metadata.json
ambient/<split>/<fmt>/*.wav       noise (optional)
```

`<fmt>` ∈ `mic` (4ch tetrahedral), `binaural` (2ch), `foa` (4ch, ACN/SN3D — AmbiX); a backbone's
`audio_format` maps `stereo`→`mic` (downmixed on load), otherwise identity. Metadata
gives `coordinates` (az/el/distance) for source and `rt60`/`volume`/`shape` for room.
Scenes (source ⊗ RIR + ambient) are synthesized on the fly, deterministic per seed.

## Citation

```bibtex
@article{chen2026sarl,
  title   = {Probing Spatial Structure in Pretrained Audio Representations},
  author  = {Chen, Chuyang and Ding, Sivan and Roman, Adrian S. and Bello, Juan P.},
  journal = {arXiv preprint arXiv:2606.05544},
  year    = {2026},
}
```

## License

MIT — see [LICENSE](LICENSE).
