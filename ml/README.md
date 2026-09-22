# EmotionAI — Machine Learning

Training and evaluation live here. Production inference lives in `backend/`.

## Phase 1 (current)

Preprocess the local FER2013 **folder** dataset (`train/`, `test/` at repo root).

### Setup (Windows PowerShell)

```powershell
cd "d:\Apna Kaam\Emotion_Detector"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r ml\requirements.txt
```

### Validate dataset only

```powershell
$env:PYTHONPATH = (Get-Location).Path
python -m ml.preprocessing.pipeline --validate-only
```

### Build manifests + NumPy arrays (train / val / test)

```powershell
$env:PYTHONPATH = (Get-Location).Path
python -m ml.preprocessing.pipeline
```

Outputs:

- `data/processed/manifest.csv`
- `data/processed/X_{train,val,test}.npy`, `y_{train,val,test}.npy`
- `data/processed/labels.json` (copy of shared mapping)
- `data/reports/dataset_validation.json`
- `data/reports/removed_records.json`
- `data/reports/preprocess_summary_latest.json`

### Manifest only (lighter disk / memory)

```powershell
python -m ml.preprocessing.pipeline --manifest-only
```

### Tests

```powershell
$env:PYTHONPATH = (Get-Location).Path
pytest ml\tests -q
```

## Class mapping

Single source of truth: `shared/labels.json`

| Index | Expression |
|------:|------------|
| 0 | Angry |
| 1 | Happy |
| 2 | Sad |
| 3 | Surprise |
| 4 | Neutral |

Excluded from v1: `disgust`, `fear`.

## Phase 2 (next, after Phase 1 verification)

CNN training (Colab GPU recommended) + evaluation + `models/emotion_cnn.keras` export.
