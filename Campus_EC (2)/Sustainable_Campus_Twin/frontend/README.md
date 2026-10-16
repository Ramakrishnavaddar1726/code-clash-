# Sustainable Campus Twin — Frontend

This folder contains the requested standalone HTML/CSS/JavaScript frontend:
- `index.html` — dashboard layout and views
- `style.css` — responsive dark navy / neon green styling
- `script.js` — navigation, demo issue injection, charts, image preview, and demo summary download

## Preview the frontend

Open `index.html` in Chrome, or open the project folder in VS Code and use the Live Server extension on `frontend/index.html`.

This is a standalone UI demo. The charts use illustrative values and the image picker previews the selected image; it does not call the Python detection model. For the Python-powered dashboard with simulated sensor analysis, AI recommendations, and PDF report, run from the project root:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The Streamlit dashboard is the existing integrated Python application. The standalone frontend will require an API layer to exchange live data with that Python backend.
