# SSL-AASIST: Voice Clone / Spoofing Detection

This repo documents reading a research paper closely enough to rebuild it —
not running someone else's code, but working from the paper's own equations,
tables, and cited references to reconstruct the architecture, training
recipe, and evaluation protocol, then comparing the results against what the
paper reports.

Reproduction of Tak, Todisco, Wang, Jung, Yamagishi, Evans, _"Automatic Speaker
Verification Spoofing and Deepfake Detection Using Wav2Vec 2.0 and Data
Augmentation"_ (Odyssey 2022, [arXiv:2202.12233](https://arxiv.org/abs/2202.12233)).

A countermeasure system (bona fide vs. spoofed speech) combining a wav2vec 2.0
XLS-R (0.3B) front-end with an AASIST graph-attention backend, trained and
evaluated on ASVspoof2019/2021.

**Demo:** [voiceprint](https://huggingface.co/spaces/eyeeeeshe/voiceprint)

**Paper:** [arXiv:2202.12233](https://arxiv.org/abs/2202.12233)

**Official authors' repo:** [TakHemlata/SSL_Anti-spoofing](https://github.com/TakHemlata/SSL_Anti-spoofing)

## Architecture

- **Front-end:** wav2vec 2.0 XLS-R (0.3B), fine-tuned jointly with the backend;
  a sinc-layer variant is included for the baseline comparison in Table 2.
- **Backend:** AASIST — residual encoder, self-attentive spectro-temporal
  aggregation, parallel graph attention modules, heterogeneous stacking graph
  attention (HS-GAL), max graph operation, node-wise readout.
- **Augmentation:** RawBoost — convolutive, impulsive, and stationary noise
  injection on the raw waveform.
- **Training:** Adam, lr 1e-6 (wav2vec2) / 1e-4 (sinc), batch size 14, 100
  epochs, weighted cross-entropy, 3 seeds.

Where this implementation diverges from a literal reading of the paper:

- The GAT/HS-GAL attention formula and RawBoost's exact noise-parameter
  ranges are specified in the paper only by citation to external work;
  `src/model/graph_modules.py` and `src/rawboost.py` implement the described
  mechanisms with documented, reasoned defaults.
- Uses HuggingFace `transformers` for the wav2vec2 XLS-R checkpoint rather
  than fairseq.
- min t-DCF uses the standard tandem-DCF formulation; matching the paper's
  published numbers exactly requires the official ASV score files.

## Repository structure

```
src/
  model/          AASIST architecture — front-ends, encoder, graph modules, aggregation
  data/           ASVspoof protocol-format dataset loader
  rawboost.py     Data augmentation
  metrics.py      EER / min t-DCF
  train.py        Training
  evaluate.py     Scoring, comparison against the paper's published tables
space/            Gradio app (Hugging Face Spaces)
```

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Data

- **ASVspoof2019 LA** (train/dev), 7.1 GB — [`LA.zip`](https://datashare.ed.ac.uk/handle/10283/3336).
- **ASVspoof2021 LA** eval + keys, needed only for evaluation —
  [speech](https://zenodo.org/records/4837263) ·
  [keys](https://www.asvspoof.org/asvspoof2021/LA-keys-full.tar.gz)

Extract to:

```
data/
  LA/
    ASVspoof2019_LA_cm_protocols/
    ASVspoof2019_LA_train/flac/
    ASVspoof2019_LA_dev/flac/
  ASVspoof2021_LA_eval/
    flac/
    keys/
```

`train.py`'s defaults point at this layout directly.

### Training

```bash
python src/train.py --config sinc_no_sa_no_da --epochs 100 --seed 1
```

`--config` selects one of the presets in `src/model/aasist.py`, covering every
row of Table 2 (front-end × self-attentive aggregation × augmentation) and
the DF-database variants from Table 3. `--subset_frac` and `--fp16` are
available for faster GPU iteration.

### Evaluation

```bash
python src/evaluate.py --config sinc_no_sa_no_da \
    --checkpoint runs/sinc_no_sa_no_da/seed1/best.pt \
    --eval_protocol data/ASVspoof2021_LA_eval/keys/trial_metadata.txt \
    --eval_dir data/ASVspoof2021_LA_eval/flac

python src/evaluate.py --compare_table table2
```

## Results

| Config                 | EER (paper) | EER (this repo) |
| ---------------------- | ----------- | --------------- |
| sinc, no SA, no DA     | 11.47%      | —               |
| wav2vec2, no SA, no DA | 6.15%       | —               |
| sinc, SA, no DA        | 8.73%       | —               |
| wav2vec2, SA, no DA    | 4.48%       | —               |
| sinc, SA, DA           | 7.65%       | —               |
| wav2vec2, SA, DA       | 0.82%       | —               |

## Demo

`space/` is a Gradio app for Hugging Face Spaces — upload or record a clip,
get a bona-fide/spoof score, adjust the decision threshold live. See
`space/README.md` for deployment.

## License / citation

Model architecture and training recipe from the cited paper; code is an
independent implementation. ASVspoof2019/2021 data used under their
respective licenses (Open Data Commons Attribution License) and not
redistributed in this repository.
