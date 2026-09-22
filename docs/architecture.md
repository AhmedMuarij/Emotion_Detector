# EmotionAI Architecture

## Goal

EmotionAI is a **facial expression recognition** platform. It classifies *visible* facial expressions into five labels (Angry, Happy, Sad, Surprise, Neutral). It does **not** claim to read internal emotions, mental health, intentions, or personality.

## Confirmed architecture

```
[Google Colab / local ML]          [Production]
FER2013 → preprocess → CNN train   Browser webcam (getUserMedia)
                ↓                           ↓
         emotion_cnn.keras          Next.js frontend (Vercel)
         metadata.json                      ↓ HTTP POST /predict
                ↓                   FastAPI backend (Render/Railway)
         copied into models/                ↓
                                   OpenCV face crop → same preprocess
                                            ↓
                                   CNN inference (model loaded once)
```

### Separation of training and inference

| Concern | Training (Colab / `ml/`) | Inference (`backend/`) |
|--------|---------------------------|-------------------------|
| Hardware | GPU preferred | CPU-friendly hosting OK for MVP |
| Data | Full FER2013, augmentation | Single face crop per request |
| Output | `.keras` + metrics plots | JSON probabilities |
| Lifecycle | Offline, infrequent | Online, low latency |
| Camera | Not used | Browser supplies frames |

The backend **never** uses `cv2.VideoCapture(0)` for the user’s laptop camera. Frames are uploaded from the browser.

### Shared contracts

- `shared/labels.json` — class indices, image size (48×48), grayscale, `/255` normalization
- `models/metadata.json` — runtime copy after training (version, metrics placeholders)

Backend and frontend must consume the same label mapping. Do not hardcode divergent label lists.

### MVP scope (intentional)

- No auth, no DB, no image persistence
- HTTP frame sampling (not WebSockets) until the baseline works
- Five FER2013 classes only

### Extension points

- Session history / analytics → Supabase later
- Auth → provider-backed sessions later
- Streaming → WebSocket after HTTP path is verified

## Repository layout

```
Emotion_Detector/
├── shared/labels.json          # single class-mapping source of truth
├── ml/                         # train / eval / preprocess (offline)
├── models/                     # exported weights + metadata (weights gitignored)
├── backend/                    # FastAPI inference (Phase 3)
├── frontend/                   # Next.js UI (Phase 4)
├── docs/                       # architecture, API, deployment
├── data/processed|reports/     # generated artifacts (gitignored)
├── train/ test/                # local FER2013 folders (gitignored)
└── .env.example
```

## Dataset note (this workspace)

This clone already contains FER2013-style **image folders** (`train/<class>/*.jpg`, `test/<class>/*.jpg`), 48×48 grayscale. Preprocessing filters to the five target classes and builds a stratified train/val split from `train/`, keeping `test/` held out.
