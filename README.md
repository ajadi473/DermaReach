# DermaReach

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Citation File](https://img.shields.io/badge/Cite-CITATION.cff-blue)](CITATION.cff)

Public companion code for the master's research project:

**Classification of Selected Skin Diseases Using Artificial Intelligence for Dark Skin Tones**  
Paul Ajadi & Nancy Woods — University of Ibadan

DermaReach trains a convolutional neural network (VGG16 transfer learning) to classify **acne**, **eczema**, and **keloids** on dark skin tones. The work addresses a known gap: most dermatology AI datasets under-represent African skin tones, which limits model usefulness in many African clinical settings.

> Abstract excerpt: [docs/ABSTRACT.md](docs/ABSTRACT.md)
> Prefer machine-readable citation: [CITATION.cff](CITATION.cff) (GitHub “Cite this repository”).

## Research summary

| Item | Detail |
|------|--------|
| Task | 3-class image classification (acne / eczema / keloids) |
| Primary dataset | 400 clinical images (acne 106, eczema 132, keloids 162) |
| After augmentation | ~2,650 training views via Keras `ImageDataGenerator` |
| Model | Frozen ImageNet VGG16 backbone + custom dense head |
| Reported overall accuracy | ~57% (first local-data baseline) |
| Strongest class | Keloids (highest precision in the study) |

This repository reproduces the training and evaluation pipeline used in the study. It does **not** redistribute the clinical image dataset (patient privacy / clinic agreements).

## Repository contents

| Path | Description |
|------|-------------|
| `train_classifier.py` | Clean, local/Colab-friendly training & evaluation script |
| `preprocess_heic.py` | Optional HEIC → JPEG converter for phone captures |
| `african_skin_disease_classification_colab.py` | Original Google Colab export (historical reference) |
| `CITATION.cff` | Citation metadata for GitHub / reference managers |
| `docs/ABSTRACT.md` | Sanitized public abstract excerpt |
| `requirements.txt` | Python dependencies |
| `data/` | Place your local class folders here (not shipped) |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data layout

Organize JPEG images as:

```text
data/
  acne/
    img001.jpg
    ...
  eczema/
    ...
  keloids/
    ...
```

If your captures are still HEIC:

```bash
python preprocess_heic.py --source /path/to/heic --destination ./data
```

Then move converted files into the class folders above.

## Train

```bash
python train_classifier.py \
  --data-dir ./data \
  --output-dir ./outputs \
  --epochs 20 \
  --batch-size 64 \
  --save-model
```

Artifacts written to `outputs/`:

- `sample_training_batch.png`
- `learning_rate_schedule.png`
- `roc_curve.png`
- `metrics.txt` (accuracy, F1, confusion matrix, classification report)
- `dermareach_vgg16.keras` (when `--save-model` is set)

## Method (short)

1. Collect anonymized clinic images of acne, eczema, and keloids on dark skin tones.
2. Convert to JPEG, resize, normalize; augment with rotations, shifts, shear, zoom, and flips.
3. Stratified train/validation split (default 75/25).
4. Train VGG16 (ImageNet weights, frozen) with a custom classification head and a stepwise learning-rate schedule.
5. Evaluate with accuracy, precision/recall/F1, confusion matrix, and ROC/AUC.

## Citation

If you use this code or refer to the study, please cite:

> Ajadi, P., & Woods, N. (2026). *Classification of Selected Skin Diseases Using Artificial Intelligence for Dark Skin Tones*. University of Ibadan. https://github.com/ajadi473/DermaReach

Or export BibTeX/APA from GitHub via **Cite this repository** (powered by `CITATION.cff`).

### Zenodo archive (for a stable thesis DOI)

1. Sign in at [zenodo.org](https://zenodo.org) with GitHub.
2. Enable the GitHub → Zenodo integration for `ajadi473/DermaReach`.
3. Create a GitHub Release (e.g. `v1.0.0`); Zenodo mints a DOI automatically.
4. Add the DOI badge to this README, for example:

```markdown
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.RECORD_ID.svg)](https://doi.org/10.5281/zenodo.RECORD_ID)
```

Update the citation with the final journal venue (and DOI) once the article is published.

## Ethics & limitations

- Clinical images are sensitive. Do not commit patient data to git.
- Reported performance (~57% overall) is a **baseline** on a small local dataset; eczema remains the hardest class due to visual overlap with other conditions.
- This software is for research and educational use. It is **not** a medical device and must not be used as a sole clinical diagnostic tool.

## License

Released under the MIT License — see [LICENSE](LICENSE).

## Contact

Paul Ajadi — ajadi.ololade@gmail.com
