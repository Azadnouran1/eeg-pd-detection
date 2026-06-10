"""
╔══════════════════════════════════════════════════════════════════╗
║   Parkinson's Disease Detection & Clinical Decision Support      ║
║   GUI Application — Streamlit Web Interface                      ║
║                                                                  ║
║   Requirements:                                                  ║
║     pip install streamlit openai torch torchvision pillow        ║
║                                                                  ║
║   Run on ACES:                                                   ║
║     streamlit run pd_streamlit_app.py --server.port 8501         ║
║             --server.address 0.0.0.0                             ║
║                                                                  ║
║   Then in a NEW terminal on your LOCAL machine:                  ║
║     ssh -L 8501:localhost:8501 u.aa346327@aces.tamu.edu          ║
║   Then open: http://localhost:8501                               ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os, json, datetime
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import streamlit as st
from openai import OpenAI

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "OPENAI_API_KEY")

MODEL_PATH = (
    "/scratch/user/u.aa346327/"
    "results - parkinson - cnn - 2/EWT/models/"
    "EWT_HC_vs_PD_groupkfold_fold5_best.pth"
)
IMAGE_SIZE = 224

# ─────────────────────────────────────────────────────────────────
# MODEL — exact architecture from iowa_residual_cnn_v7.py
# ─────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1    = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1      = nn.BatchNorm2d(out_channels)
        self.relu     = nn.ReLU(inplace=True)
        self.conv2    = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2      = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResidualCustomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.res_block1 = ResidualBlock(32, 32)
        self.pool1      = nn.MaxPool2d(2)
        self.drop1      = nn.Dropout2d(0.1)

        self.res_block2 = ResidualBlock(32, 64)
        self.pool2      = nn.MaxPool2d(2)
        self.drop2      = nn.Dropout2d(0.1)

        self.res_block3 = ResidualBlock(64, 128)
        self.pool3      = nn.MaxPool2d(2)
        self.drop3      = nn.Dropout2d(0.1)

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.drop1(self.pool1(self.res_block1(x)))
        x = self.drop2(self.pool2(self.res_block2(x)))
        x = self.drop3(self.pool3(self.res_block3(x)))
        x = self.gap(x)
        return self.classifier(x)


inference_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = ResidualCustomCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model.to(device), device


def classify_image(uploaded_file):
    model, device = load_model()
    image  = Image.open(uploaded_file).convert("L")
    tensor = inference_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logit   = model(tensor).squeeze()
        pd_prob = torch.sigmoid(logit).item()
    label      = "PD" if pd_prob >= 0.5 else "HC"
    confidence = pd_prob if label == "PD" else 1.0 - pd_prob
    return label, round(confidence, 4)


# ─────────────────────────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are a clinical decision support assistant specialized in Parkinson's Disease management.
Follow MDS, AAN, and NICE PD guidelines. Generate a personalized treatment plan.
RULES: tailor to demographics, flag interactions, give dose RANGES only, include non-pharma.
Respond ONLY in valid JSON — no preamble, no markdown.

JSON STRUCTURE:
{
  "patient_summary": "...",
  "disease_stage": "Early | Moderate | Advanced",
  "pharmacological_treatment": {
    "first_line": {"drug":"...","class":"...","dose_range":"...","rationale":"..."},
    "adjunct_options": [{"drug":"...","class":"...","dose_range":"...","indication":"..."}]
  },
  "contraindications_and_interactions": ["..."],
  "non_pharmacological": ["..."],
  "lifestyle_recommendations": ["..."],
  "monitoring_plan": ["..."],
  "disclaimer": "..."
}
"""


def get_treatment_plan(patient):
    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt = f"""
Patient classified as PD via EWT CNN (GroupKFold Fold 5).
Demographics:
- ID: {patient.get('subject_id')}  Age: {patient.get('age')}  Sex: {patient.get('sex')}
- Height/Weight: {patient.get('height_cm')}cm / {patient.get('weight_kg')}kg
- BP: {patient.get('blood_pressure')}  Blood Type: {patient.get('blood_type')}
- Disease Duration: {patient.get('disease_duration_years')} yrs
- Dominant Symptom: {patient.get('dominant_symptom')}
- Cognitive Status: {patient.get('cognitive_status')}
- Comorbidities: {patient.get('comorbidities')}
- Medications: {patient.get('current_medications')}
- Allergies: {patient.get('allergies')}
- UPDRS Motor: {patient.get('updrs_motor')}
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":prompt}],
        temperature=0.3, max_tokens=1500,
        response_format={"type":"json_object"}
    )
    try:
        return json.loads(response.choices[0].message.content)
    except:
        return {"error": "Invalid JSON from LLM"}


# ─────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────────────────────────

st.set_page_config(page_title="PD Detection & Treatment", page_icon="🧠", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family:'IBM Plex Sans',sans-serif; background:#0d1117; color:#e6edf3; }
.main { background:#0d1117; }
.stButton>button { background:#1f6feb; color:white; border:none; border-radius:6px; font-weight:600; padding:10px 24px; }
.stButton>button:hover { background:#388bfd; }
.block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div style="background:linear-gradient(135deg,#161b22,#1a2332);border:1px solid #30363d;
border-top:3px solid #58a6ff;border-radius:10px;padding:24px 32px;margin-bottom:24px;">
<h1 style="font-family:'IBM Plex Mono',monospace;font-size:1.5em;color:#58a6ff;margin:0 0 6px 0;">
🧠 Parkinson's Disease Detection & Clinical Decision Support</h1>
<p style="color:#8b949e;margin:0;font-size:0.9em;">
EWT Spectrogram · Custom Residual CNN · GroupKFold Fold 5 · GPT-4o Treatment Planning</p>
<p style="color:#6e7681;margin:4px 0 0 0;font-size:0.78em;">
University of North Texas — Information Science PhD Research</p>
</div>
""", unsafe_allow_html=True)

# Session state
for key in ["classification","confidence","show_form","plan","patient"]:
    if key not in st.session_state:
        st.session_state[key] = None
if "show_form" not in st.session_state:
    st.session_state.show_form = False

# ── STEP 1 ────────────────────────────────────────────────────────
st.markdown("#### 📤 Step 1 — Upload & Classify EWT Spectrogram")
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    uploaded = st.file_uploader("Upload EWT image", type=["png","jpg","jpeg"], label_visibility="collapsed")
    if uploaded:
        st.image(uploaded, caption="Uploaded EWT Spectrogram", use_column_width=True)
    if st.button("🔍  Classify Image", type="primary", use_container_width=True):
        if not uploaded:
            st.warning("Please upload an EWT image first.")
        else:
            with st.spinner("Running CNN inference..."):
                lbl, conf = classify_image(uploaded)
            st.session_state.classification = lbl
            st.session_state.confidence     = conf
            st.session_state.show_form      = False
            st.session_state.plan           = None
            st.rerun()

with col2:
    if st.session_state.classification == "HC":
        st.markdown(f"""
        <div style="background:#0d1f0d;border:1px solid #3fb950;border-radius:10px;
        padding:32px;text-align:center;margin-top:12px;">
        <h2 style="color:#3fb950;margin:0 0 8px 0;">✅ HEALTHY CONTROL</h2>
        <p style="color:#8b949e;margin:0;">Confidence: <strong style="color:#3fb950">
        {st.session_state.confidence:.1%}</strong></p>
        <p style="color:#8b949e;margin:10px 0 0 0;font-size:0.9em;">
        No treatment recommendation required.</p></div>
        """, unsafe_allow_html=True)

    elif st.session_state.classification == "PD":
        st.markdown(f"""
        <div style="background:#1f0d0d;border:1px solid #f85149;border-radius:10px;
        padding:32px;text-align:center;margin-top:12px;">
        <h2 style="color:#f85149;margin:0 0 8px 0;">⚠️ PARKINSON'S DISEASE</h2>
        <p style="color:#8b949e;margin:0;">Confidence: <strong style="color:#f85149">
        {st.session_state.confidence:.1%}</strong></p>
        <p style="color:#8b949e;margin:10px 0 0 0;font-size:0.9em;">
        Fill patient demographics below to generate a treatment plan.</p></div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📋  Proceed to Demographics Form →", use_container_width=True):
            st.session_state.show_form = True
            st.rerun()

# ── STEP 2 — Demographics Form ────────────────────────────────────
if st.session_state.show_form:
    st.divider()
    st.markdown("#### 📋 Step 2 — Patient Demographic Information")

    with st.form("demographics"):
        c1, c2, c3 = st.columns(3)
        with c1:
            subject_id = st.text_input("Patient ID", value="PATIENT_001")
            age        = st.number_input("Age (years)", 18, 120, 65)
            sex        = st.selectbox("Sex", ["Male","Female","Other"])
        with c2:
            height_cm  = st.number_input("Height (cm)", 100, 220, 170)
            weight_kg  = st.number_input("Weight (kg)", 30, 200, 75)
            blood_type = st.selectbox("Blood Type", ["A+","A-","B+","B-","AB+","AB-","O+","O-","Unknown"])
        with c3:
            blood_pressure   = st.text_input("Blood Pressure (mmHg)", "130/85")
            disease_duration = st.number_input("Disease Duration (years)", 0, 40, 2)
            updrs_motor      = st.number_input("UPDRS Motor Score (0=unknown)", 0, 132, 0)

        dominant_symptom = st.text_area("Dominant Symptom(s)",
            placeholder="e.g. Resting tremor right hand, bradykinesia", height=75)
        cognitive_status = st.text_input("Cognitive Status",
            placeholder="e.g. Normal (MoCA: 28/30)  or  Mild impairment (MoCA: 23/30)")
        comorbidities    = st.text_area("Comorbidities",
            placeholder="e.g. Hypertension, Type 2 Diabetes, Depression", height=75)
        current_meds     = st.text_area("Current Medications",
            placeholder="e.g. Metformin 1000mg, Lisinopril 10mg", height=75)
        allergies        = st.text_input("Known Allergies", placeholder="e.g. Penicillin")

        submitted = st.form_submit_button(
            "⚕️  Generate Treatment Plan", type="primary", use_container_width=True)

    if submitted:
        patient = {
            "subject_id": subject_id, "age": age, "sex": sex,
            "height_cm": height_cm, "weight_kg": weight_kg,
            "blood_pressure": blood_pressure, "blood_type": blood_type,
            "disease_duration_years": disease_duration,
            "dominant_symptom": dominant_symptom or "Not specified",
            "cognitive_status": cognitive_status  or "Not assessed",
            "comorbidities":    comorbidities     or "None reported",
            "current_medications": current_meds   or "None reported",
            "allergies":        allergies         or "None known",
            "updrs_motor":      updrs_motor if updrs_motor > 0 else "N/A",
        }
        with st.spinner("Generating treatment plan via GPT-4o..."):
            plan = get_treatment_plan(patient)
        st.session_state.plan    = plan
        st.session_state.patient = patient
        st.rerun()

# ── STEP 3 — Treatment Plan Output ───────────────────────────────
if st.session_state.plan:
    plan    = st.session_state.plan
    patient = st.session_state.patient or {}

    st.divider()
    st.markdown(f"#### ⚕️ Treatment Plan — {patient.get('subject_id','N/A')} | {plan.get('disease_stage','N/A')} Stage")
    st.caption(plan.get("patient_summary",""))

    if "error" in plan:
        st.error(f"LLM error: {plan['error']}")
    else:
        pt  = plan.get("pharmacological_treatment", {})
        fl  = pt.get("first_line", {})
        adj = pt.get("adjunct_options", [])

        col_l, col_r = st.columns(2, gap="large")

        with col_l:
            st.markdown("**💊 First-Line Treatment**")
            st.markdown(f"""
            <div style="background:#0d1117;border:1px solid #21262d;border-left:3px solid #3fb950;
            border-radius:6px;padding:14px 16px;margin-bottom:12px;">
            <strong style="font-size:1.05em;">{fl.get('drug','N/A')}</strong>
            <span style="color:#8b949e;font-size:0.85em;margin-left:8px;">{fl.get('class','')}</span><br>
            <span style="color:#3fb950;">📏 {fl.get('dose_range','N/A')}</span><br>
            <span style="color:#8b949e;font-size:0.88em;">{fl.get('rationale','')}</span>
            </div>""", unsafe_allow_html=True)

            if adj:
                st.markdown("**➕ Adjunct Options**")
                for a in adj:
                    st.markdown(f"""
                    <div style="background:#0d1117;border:1px solid #21262d;border-left:3px solid #58a6ff;
                    border-radius:6px;padding:10px 14px;margin-bottom:8px;">
                    <strong>{a.get('drug','')}</strong>
                    <span style="color:#8b949e;font-size:0.82em;margin-left:6px;">{a.get('class','')}</span><br>
                    <span style="color:#3fb950;font-size:0.85em;">📏 {a.get('dose_range','')}</span><br>
                    <span style="color:#8b949e;font-size:0.85em;">{a.get('indication','')}</span>
                    </div>""", unsafe_allow_html=True)

            ci = plan.get("contraindications_and_interactions", [])
            if ci:
                st.markdown("**⚠️ Contraindications & Interactions**")
                for c in ci:
                    st.markdown(f"""
                    <div style="background:#1f1200;border-left:3px solid #d29922;border-radius:4px;
                    padding:8px 12px;margin-bottom:6px;color:#e3b341;font-size:0.9em;">⚠️ {c}</div>
                    """, unsafe_allow_html=True)

        with col_r:
            np_ = plan.get("non_pharmacological", [])
            if np_:
                st.markdown("**🏃 Non-Pharmacological**")
                for item in np_: st.markdown(f"- {item}")

            ls = plan.get("lifestyle_recommendations", [])
            if ls:
                st.markdown("**🌿 Lifestyle**")
                for item in ls: st.markdown(f"- {item}")

            mon = plan.get("monitoring_plan", [])
            if mon:
                st.markdown("**📅 Monitoring Plan**")
                for item in mon: st.markdown(f"- {item}")

        st.markdown(f"""
        <div style="background:#161b22;border:1px dashed #30363d;border-radius:6px;
        padding:12px 16px;color:#8b949e;font-size:0.82em;font-style:italic;margin-top:16px;">
        ⚕️ {plan.get('disclaimer','')}
        </div>""", unsafe_allow_html=True)

        ts     = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output = {"patient": patient, "treatment_plan": plan, "generated_at": ts}
        st.download_button(
            "💾 Download Treatment Plan (JSON)",
            data=json.dumps(output, indent=2),
            file_name=f"treatment_{patient.get('subject_id','patient')}_{ts}.json",
            mime="application/json",
            use_container_width=True
        )
        st.caption(f"*Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
