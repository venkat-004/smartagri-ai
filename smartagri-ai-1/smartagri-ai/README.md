# SmartAgri AI

AI-powered multilingual farming assistant — Flask backend, Bootstrap 5
frontend. Supports English, Telugu (తెలుగు), Tamil (தமிழ்), and Hindi
(हिन्दी).

## Features

- **Language gate** on first visit; the whole UI (nav, labels, alerts,
  voice assistant) switches language client-side, stored in
  `localStorage`.
- **Dashboard** — village, crop, soil moisture, temperature, humidity,
  with a rule-based irrigation alert. Saved to SQLite via `/api/field`.
- **Leaf disease detection** — upload a photo, get a simulated AI
  reading (condition, confidence %, treatment). See `model/README.md`
  for wiring up a real trained model.
- **Weather** — live conditions from the free
  [Open-Meteo](https://open-meteo.com) API (no key required), with
  simple farming advice generated from temperature / rain / wind.
- **Voice assistant** — uses the browser's Web Speech API for
  speech-to-text and text-to-speech in all four languages; a small
  keyword-matched backend (`/api/voice-query`) supplies the answer.

## Project structure

```
smartagri-ai/
  app.py                 Flask app, routes, REST API, SQLite models
  requirements.txt
  Procfile                gunicorn entry point for Render
  templates/
    base.html              shared header / nav / footer
    index.html              language selection screen
    dashboard.html
    disease.html
    weather.html
    voice.html
  static/
    css/style.css
    js/i18n.js               translation dictionary + language switching
  model/
    README.md               how to plug in a real trained model
```

## Run locally

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Visit `http://localhost:5000`.

The SQLite database (`smartagri.db`) is created automatically on first
run.

## Deploy to Render

1. Push this project to a GitHub repository.
2. In Render, create a **New Web Service** from that repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app` (already set in `Procfile`, so
   Render should pick it up automatically).
5. No environment variables are required — the weather API is free and
   keyless.

## Notes on this prototype

- Disease detection is **simulated**, not a trained model (see
  `model/README.md` for next steps).
- The voice assistant's "AI" answers come from a small keyword lookup,
  not a language model — swap `answer_voice_question()` in `app.py`
  for a real LLM/NLU call for production use.
- Weather and geocoding both come from Open-Meteo's live, free API.
