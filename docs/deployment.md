# EmotionAI Deployment Guide

## Prerequisites

- Trained model: `models/emotion_cnn.keras` + `models/metadata.json`
- GitHub repository with the full project committed
- Vercel account (free)
- Render account (free)

---

## Stage 1 — Local Verification

Verify the model loads and `/predict` works before deploying:

```powershell
# From repo root, venv active
$env:PYTHONPATH = "d:\Apna Kaam\Emotion_Detector\backend"
$env:MODEL_PATH = "..\models\emotion_cnn.keras"
$env:MODEL_METADATA_PATH = "..\models\metadata.json"
$env:CORS_ORIGINS = "http://localhost:3000"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Test endpoints:
```powershell
curl http://localhost:8000/health
curl http://localhost:8000/model-info
```

---

## Stage 2 — Deploy Backend to Render

### 2.1 Commit model files

Option A — Commit directly (model < 100 MB):
```bash
git add models/emotion_cnn.keras models/metadata.json
git commit -m "Add trained model v0.1.0"
git push
```

Option B — Git LFS (if model > 100 MB):
```bash
git lfs install
git lfs track "*.keras"
git add .gitattributes models/emotion_cnn.keras models/metadata.json
git commit -m "Add trained model via LFS"
git push
```

### 2.2 Create Render Web Service

1. Go to [render.com](https://render.com) → New → Web Service
2. Connect your GitHub repository
3. Set the following:

| Setting | Value |
|---------|-------|
| Root Directory | `backend` |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `uvicorn app.main:app --host 0.0.0.0 --port $PORT` |
| Health Check Path | `/health` |

4. Add environment variables:

| Key | Value |
|-----|-------|
| `CORS_ORIGINS` | `https://your-app.vercel.app` (fill after Vercel deploy) |
| `MODEL_PATH` | `../models/emotion_cnn.keras` |
| `MODEL_METADATA_PATH` | `../models/metadata.json` |
| `MAX_UPLOAD_BYTES` | `5242880` |

5. Deploy. First deploy takes 2–5 minutes.
6. Test: `curl https://your-render-url.onrender.com/health`

> **Cold starts**: Free Render tier sleeps after 15 minutes of inactivity.
> First request after sleep takes 30–60 seconds (model must load into memory).
> This is acceptable for a portfolio project. Upgrade to a paid plan for production.

---

## Stage 3 — Deploy Frontend to Vercel

### 3.1 Push frontend to GitHub (already in repo root)

Vercel auto-detects Next.js when the `frontend/` directory has `package.json`.

### 3.2 Create Vercel Project

1. Go to [vercel.com](https://vercel.com) → New Project → Import GitHub repo
2. Set **Root Directory** to `frontend`
3. Framework: **Next.js** (auto-detected)
4. Add environment variable:

| Key | Value |
|-----|-------|
| `NEXT_PUBLIC_API_BASE_URL` | `https://your-render-url.onrender.com` |

5. Deploy.

### 3.3 Update CORS on Render

Once Vercel gives you a deployment URL (e.g. `https://emotion-ai.vercel.app`):

1. Go to Render → Your Service → Environment
2. Update `CORS_ORIGINS` to your Vercel URL
3. Redeploy the backend service

---

## Stage 4 — Production Verification Checklist

- [ ] `https://your-app.vercel.app` loads in browser
- [ ] No HTTPS mixed-content errors in browser console
- [ ] `/health` returns `{"status":"ok","model_loaded":true,...}`
- [ ] Browser camera permission prompt appears when clicking Start Camera
- [ ] Predictions appear within a few seconds
- [ ] Stop Camera clears the camera indicator light
- [ ] Backend error is shown gracefully when API is unavailable
- [ ] Mobile layout looks correct (single column stack)

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `503` on `/predict` | Model not loaded | Check Render logs for model path error |
| CORS error in browser | `CORS_ORIGINS` mismatch | Update env var on Render and redeploy |
| Camera not working | HTTPS required for `getUserMedia` | Ensure Vercel uses HTTPS (it does by default) |
| Cold start timeout | Free Render plan | Accept it or upgrade; add retry logic |
| `/predict` returns `no face detected` | Poor lighting or small face | Move closer to camera |
