"""AI-Powered Sustainable Campus - Digital Twin dashboard.  Run: streamlit run app.py"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import genai
import report
from detection import (CO2_KG_PER_KWH, STEP_H, TARIFF_INR_PER_KWH, build_baseline,
                       building_status, campus_kpis, detect_issues)
from simulator import BUILDINGS, FAULT_TYPES, generate_campus

st.set_page_config(page_title="Sustainable Campus Twin", page_icon="🌿", layout="wide")


# --------------------------- UI / frontend theme
st.markdown("""
<style>
:root { color-scheme: dark; }
.stApp { background: radial-gradient(ellipse at 15% 0%, #172b3b 0%, #0a111b 48%, #070c13 100%); color: #e8f2f7; }
[data-testid="stHeader"] { background: rgba(7,12,19,.78); }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#101d2b 0%,#0b131e 100%); border-right: 1px solid #24394a; }
[data-testid="stSidebar"] * { color: #e5edf4; }
h1,h2,h3 { color: #f2f8fc !important; letter-spacing: -.025em; }
.stCaption, [data-testid="stCaptionContainer"] { color: #9fb4c4 !important; }
[data-testid="stMetric"] { background: linear-gradient(145deg,rgba(20,39,54,.94),rgba(12,24,35,.94)); border: 1px solid #294458; border-radius: 15px; padding: 16px 18px; box-shadow: 0 8px 24px rgba(0,0,0,.15); }
[data-testid="stMetricLabel"] { color: #9bb5c7 !important; }
[data-testid="stMetricValue"] { color: #66e3b4 !important; font-weight: 750; }
.stButton > button, .stDownloadButton > button { border-radius: 10px; border: 1px solid #2bdbab; background: linear-gradient(100deg,#16a779,#087e87); color: white; font-weight: 700; transition: .2s; }
.stButton > button:hover, .stDownloadButton > button:hover { border-color: #7ef4d0; color: white; transform: translateY(-1px); }
.stTabs [data-baseweb="tab-list"] { gap: 8px; border-bottom: 1px solid #263e50; }
.stTabs [data-baseweb="tab"] { background: #111f2c; border: 1px solid #253b4b; border-radius: 10px 10px 0 0; padding: 10px 15px; color: #b9cad6; }
.stTabs [aria-selected="true"] { background: #153b3c !important; color: #7cf4c8 !important; border-color: #2bdbab !important; }
[data-testid="stAlert"] { border-radius: 12px; }
[data-testid="stFileUploader"] { background: rgba(18,35,49,.7); border: 1px dashed #3e6c7d; border-radius: 14px; padding: 10px; }
[data-testid="stPlotlyChart"] { background: rgba(13,25,36,.65); border: 1px solid #20394a; border-radius: 14px; padding: 8px; }
input, textarea, [data-baseweb="select"] > div { background-color: #101e2a !important; }
hr { border-color: #253b4b !important; }
.hero { padding: 22px 26px; border-radius: 18px; margin: 2px 0 20px 0; background: linear-gradient(110deg,rgba(18,56,63,.95),rgba(20,34,54,.92) 60%,rgba(47,29,73,.8)); border: 1px solid #2c5d63; box-shadow: 0 12px 32px rgba(0,0,0,.18); }
.hero-kicker { color:#75edc7; text-transform:uppercase; font-size: .75rem; font-weight:800; letter-spacing:.16em; }
.hero-title { font-size:2rem; font-weight:850; color:#f3fbff; margin-top:5px; }
.hero-sub { color:#b7ccd7; margin-top:5px; }
.small-note { color:#9bb4c4; font-size:.82rem; }
@media(max-width:800px){ .hero-title{font-size:1.5rem;} }
</style>
""", unsafe_allow_html=True)

COLORS = {"green": "#2ecc71", "yellow": "#f1c40f", "red": "#e74c3c"}

# ---------------------------------------------------------------- state / data
if "faults" not in st.session_state:
    st.session_state.faults = {}
if "ai_text" not in st.session_state:
    st.session_state.ai_text = ""
if "end" not in st.session_state:
    st.session_state.end = str(pd.Timestamp.now().floor("15min"))


@st.cache_data(show_spinner=False)
def load(faults_key: tuple, end: str) -> pd.DataFrame:
    return generate_campus(end=end, days=8, faults={b: list(f) for b, f in faults_key})


@st.cache_resource(show_spinner=False)
def get_baseline(end: str):
    df = load((), end)
    return build_baseline(df[df["time"] < pd.Timestamp(end) - pd.Timedelta(hours=24)])


end = st.session_state.end
fkey = tuple(sorted((b, tuple(sorted(f))) for b, f in st.session_state.faults.items() if f))
df = load(fkey, end)
cut = pd.Timestamp(end) - pd.Timedelta(hours=24)
live = df[df["time"] >= cut]
issues = detect_issues(live, get_baseline(end))
kpis = campus_kpis(live, issues)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🌿 Campus Twin")
    st.caption("Data source: **simulated** sensor streams (swap with MQTT/IoT feeds).")
    st.subheader("⚡ Inject a fault")
    fb = st.selectbox("Building", list(BUILDINGS))
    ft = st.selectbox("Fault", list(FAULT_TYPES), format_func=FAULT_TYPES.get)
    c1, c2 = st.columns(2)
    if c1.button("Inject", type="primary", width="stretch"):
        cur = set(st.session_state.faults.get(fb, []))
        cur.add(ft)
        st.session_state.faults[fb] = sorted(cur)
        st.session_state.ai_text = ""
        st.rerun()
    if c2.button("Reset all", width="stretch"):
        st.session_state.faults = {}
        st.session_state.ai_text = ""
        st.rerun()
    st.divider()
    st.caption("🤖 GenAI: " + ("Claude connected" if genai.llm_available()
                               else "offline template mode (set ANTHROPIC_API_KEY)"))

st.markdown(f"""
<div class="hero">
  <div class="hero-kicker">🌿 Sustainable Campus Intelligence</div>
  <div class="hero-title">AI-Powered Campus Twin</div>
  <div class="hero-sub">Monitor resources · Detect anomalies · Take action · Measure impact</div>
  <div class="small-note">Data window: last 24 hours ending {end[:16]} · Simulated sensor data for demo</div>
</div>
""", unsafe_allow_html=True)

k = st.columns(5)
k[0].metric("Energy (24h)", f"{kpis['energy_kwh_24h']:,} kWh")
k[1].metric("Water (24h)", f"{kpis['water_kl_24h']} kL")
k[2].metric("CO₂ (24h)", f"{kpis['co2_kg_24h']:,} kg")
k[3].metric("Active issues", kpis["active_issues"])
k[4].metric("Loss if unresolved", f"₹{kpis['monthly_loss_inr']:,}/mo")

tab_twin, tab_an, tab_alert, tab_wi, tab_rep, tab_cv = st.tabs(
    ["🏫 Digital Twin", "📈 Analytics", "🚨 Alerts & AI", "🔮 What-if", "📄 Report", "👁️ Vision"])

# ---------------------------------------------------------------- twin map
with tab_twin:
    last = live.sort_values("time").groupby("building").tail(1).set_index("building")
    xs, ys, cols, names, hover = [], [], [], [], []
    for b, cfg in BUILDINGS.items():
        r = last.loc[b]
        stt = building_status(issues, b)
        mine = [f"⚠ {i['type']}" for i in issues if i["building"] == b]
        xs.append(cfg["pos"][0]); ys.append(cfg["pos"][1]); cols.append(COLORS[stt]); names.append(b)
        hover.append(f"<b>{b}</b><br>Energy: {r.energy_kw:.1f} kW<br>Water: {r.water_lpm:.1f} L/min"
                     f"<br>Occupancy: {int(r.occupancy)}<br>Bin: {r.bin_pct:.0f}%<br>"
                     + ("<br>".join(mine) if mine else "✔ All normal"))
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0.2, y0=0.2, x1=6.3, y1=4.0, line=dict(color="#3a4a5a", width=2, dash="dot"))
    fig.add_shape(type="line", x0=0.2, y0=2.0, x1=6.3, y1=2.0, line=dict(color="#556", width=6))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers+text", text=names, textposition="bottom center",
                             textfont=dict(color="white", size=14), hovertext=hover, hoverinfo="text",
                             marker=dict(symbol="square", size=70, color=cols,
                                         line=dict(color="white", width=2))))
    fig.update_xaxes(visible=False, range=[0, 6.5])
    fig.update_yaxes(visible=False, range=[0, 4.3])
    fig.update_layout(height=470, margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor="#0f1722",
                      paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
    st.plotly_chart(fig, width="stretch")
    st.caption("🟢 normal   🟡 warning   🔴 critical  -  hover a building for live readings. "
               "Use **Inject a fault** in the sidebar to see the twin react.")

# ---------------------------------------------------------------- analytics
with tab_an:
    sel = st.selectbox("Building", list(BUILDINGS), key="an_b")
    d = live[live["building"] == sel]
    mine = [i for i in issues if i["building"] == sel]
    for col, title, unit in [("energy_kw", "Energy", "kW"), ("water_lpm", "Water flow", "L/min"),
                             ("bin_pct", "Bin fill level", "%")]:
        f = px.line(d, x="time", y=col, title=f"{title} ({unit})")
        for i in mine:
            f.add_vrect(x0=i["since"], x1=d["time"].max(), fillcolor="red", opacity=0.12, line_width=0)
        f.update_layout(height=250, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(f, width="stretch")

# ---------------------------------------------------------------- alerts + AI
with tab_alert:
    if not issues:
        st.success("No active issues. Inject a fault from the sidebar to demo the full loop.")
    for i in issues:
        box = st.error if i["severity"] == "high" else st.warning
        box(f"**{i['type']} - {i['building']}**  ({i['severity'].upper()}, since {i['since'][11:16]})\n\n"
            f"{i['detail']}  \nEst. loss: **₹{i['monthly_cost_inr']:,}/month**")
    if st.button("🤖 Generate AI recommendations", type="primary"):
        with st.spinner("Analysing..."):
            st.session_state.ai_text = genai.recommendations(issues, kpis)
    if st.session_state.ai_text:
        st.markdown(st.session_state.ai_text)
    st.divider()
    q = st.text_input("Ask the campus assistant", placeholder="Which building is losing the most money?")
    if q:
        with st.spinner("Thinking..."):
            st.write(genai.ask(q, issues, kpis))

# ---------------------------------------------------------------- what-if
with tab_wi:
    st.subheader("What-if savings simulator")
    idle_kwh_day = float(live.loc[live["occupancy"] == 0, "energy_kw"].sum() * STEP_H)
    total_kwh_day = kpis["energy_kwh_24h"]
    a = st.slider("Cut energy used in empty rooms by (%)", 0, 100, 40)
    b = st.slider("Raise AC set-point by (°C)", 0, 4, 1)
    c = st.checkbox("Fix all detected water leaks", value=True)
    save_idle = a / 100 * idle_kwh_day * 30
    save_ac = b * 0.06 * 0.4 * total_kwh_day * 30      # ~6%/°C on an assumed 40% AC share
    save_leak = sum(i["monthly_cost_inr"] for i in issues if i["type"] == "Water leakage") if c else 0
    kwh_saved = save_idle + save_ac
    inr = kwh_saved * TARIFF_INR_PER_KWH + save_leak
    w = st.columns(3)
    w[0].metric("Savings / month", f"₹{inr:,.0f}")
    w[1].metric("Energy saved", f"{kwh_saved:,.0f} kWh")
    w[2].metric("CO₂ avoided", f"{kwh_saved * CO2_KG_PER_KWH:,.0f} kg")
    st.caption("Assumptions: ₹8/kWh, ~6% AC energy saving per °C on an assumed 40% AC share, "
               "₹30 per kL water, 0.82 kg CO₂/kWh.")

# ---------------------------------------------------------------- report
with tab_rep:
    st.subheader("Sustainability report")
    if st.button("Build PDF report"):
        if not st.session_state.ai_text:
            st.session_state.ai_text = genai.recommendations(issues, kpis)
        st.session_state.pdf = report.build_pdf(issues, kpis, st.session_state.ai_text)
    if st.session_state.get("pdf"):
        st.download_button("⬇️ Download PDF", st.session_state.pdf,
                           file_name="campus_sustainability_report.pdf", mime="application/pdf")

# ---------------------------------------------------------------- vision
with tab_cv:
    st.subheader("Computer Vision (YOLOv8)")
    st.caption("Upload a frame for analysis. People counting uses a pretrained COCO model; waste detection requires your trained models/waste.pt weights. Images are not saved by this app.")
    mode = st.radio("Task", ["People count (energy wastage)", "Waste / overflowing bin"], horizontal=True)
    up = st.file_uploader("Upload a camera frame", type=["jpg", "jpeg", "png"])
    if up:
        try:
            import vision
            if mode.startswith("People"):
                n, img = vision.count_people(up.getvalue())
                st.image(img, caption=f"{n} person(s) detected", width="stretch")
                lights = st.toggle("Lights / AC are ON in this room", value=True)
                if n == 0 and lights:
                    st.error("⚠ Energy wastage: room empty but lights/AC on.")
                else:
                    st.success("Room usage looks normal.")
            elif vision.waste_available():
                found, img = vision.detect_waste(up.getvalue())
                st.image(img, width="stretch")
                st.write(found or "No waste detected.")
            else:
                st.warning("Waste model not installed yet. Add a waste-trained YOLO model at models/waste.pt. The default YOLO model detects people and common objects, not reliable waste classes.")
        except ImportError:
            st.error("Install the vision dependency:  pip install ultralytics")
