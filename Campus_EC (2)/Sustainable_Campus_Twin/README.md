# AI-Powered Sustainable Campus - Digital Twin

Simulated sensors + Computer Vision + anomaly detection -> Digital Twin -> Generative AI recommendations and PDF reports.

```
simulator.py  ->  detection.py  ->  app.py (twin, alerts, what-if)  ->  genai.py  ->  report.py (PDF)
 (fake IoT)       (rules + ML)        vision.py (YOLOv8, optional)
```

## Run it (5 minutes)
```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
set ANTHROPIC_API_KEY=your_key_here              # optional on Windows; without it a template is used
streamlit run app.py
```
Optional vision: `pip install ultralytics` (people counting works out of the box).

## Demo script (2 minutes)
1. Open **Digital Twin**: all buildings green.
2. Sidebar: inject **Water leakage -> Library**. The building turns red, an alert appears.
3. Inject **Energy wastage -> Block A** and **Bin overflow -> Canteen**.
4. **Alerts & AI**: click *Generate AI recommendations*.
5. **What-if**: move the sliders to show ₹, kWh and CO2 saved.
6. **Report**: build and download the PDF.
7. **Vision**: upload an empty-classroom photo, 0 people + lights on = wastage flagged.

## What each file does
| File | Role |
|---|---|
| simulator.py | 5 buildings, 8 days of 5-min data (energy, water, occupancy, lights, bin level) + fault injection |
| detection.py | Leak (z-score vs hourly norm), energy wastage (empty + lights on), bin overflow, Isolation Forest for unusual patterns |
| genai.py | Claude recommendations, report text, Q&A chat (template fallback offline) |
| report.py | PDF report with ReportLab |
| vision.py | YOLOv8 people counting and waste detection |
| app.py | Streamlit dashboard |

## Train the waste-detection model (Google Colab, free GPU)
```python
!pip install ultralytics roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_KEY")
ds = rf.workspace("WORKSPACE").project("PROJECT").version(1).download("yolov8")  # pick a garbage/overflow dataset on Roboflow Universe
from ultralytics import YOLO
YOLO("yolov8n.pt").train(data=ds.location + "/data.yaml", epochs=40, imgsz=640)
# download runs/detect/train/weights/best.pt -> save as models/waste.pt
```

## Going real
Replace `generate_campus()` with a function that reads MQTT/HTTP messages with the same columns
(time, building, energy_kw, water_lpm, occupancy, lights_on, bin_pct). Nothing else changes.
CCTV frames go through `vision.count_people()` to fill `occupancy`.

## Be honest in your presentation
- Sensor data is simulated; the pipeline is built to accept real feeds.
- Cost and CO2 figures use stated assumptions (₹8/kWh, ₹30/kL, 0.82 kg CO2/kWh).
- Privacy: only counts are used, no faces or images stored.
