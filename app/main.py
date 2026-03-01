# ── main.py — VitalWatch FastAPI Backend ──────────────────────────────────────
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib
import json
import torch
import torch.nn as nn
import shap
import os
import anthropic
from dotenv import load_dotenv
from parser import parse_file, FEATURE_COLS

load_dotenv()

app = FastAPI(
    title       = 'VitalWatch API',
    description = 'Early Sepsis Detection — TECO + XGBoost + AI Agent',
    version     = '1.0.0'
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ['*'],
    allow_credentials = True,
    allow_methods     = ['*'],
    allow_headers     = ['*'],
)

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(BASE_DIR, 'models')
MAX_SEQ_LEN = 168


# ── TECO Architecture ─────────────────────────────────────────────────────────
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 dilation, dropout=0.3):
        super().__init__()
        padding       = (kernel_size - 1) * dilation
        self.conv1    = nn.Conv1d(in_channels, out_channels, kernel_size,
                                  dilation=dilation, padding=padding)
        self.conv2    = nn.Conv1d(out_channels, out_channels, kernel_size,
                                  dilation=dilation, padding=padding)
        self.bn1      = nn.BatchNorm1d(out_channels)
        self.bn2      = nn.BatchNorm1d(out_channels)
        self.dropout  = nn.Dropout(dropout)
        self.relu     = nn.ReLU()
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        residual = self.residual(x)
        out = self.conv1(x)
        out = out[:, :, :-self.conv1.padding[0]]
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = out[:, :, :-self.conv2.padding[0]]
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        return self.relu(out + residual)


class TECOModel(nn.Module):
    def __init__(self, input_size, num_channels=[64, 128, 256, 512],
                 kernel_size=3, dropout=0.3, num_heads=8):
        super().__init__()
        self.input_proj = nn.Linear(input_size, num_channels[0])
        layers = []
        for i in range(len(num_channels)):
            in_ch  = num_channels[i-1] if i > 0 else num_channels[0]
            out_ch = num_channels[i]
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size,
                                        dilation=2**i, dropout=dropout))
        self.tcn        = nn.Sequential(*layers)
        self.attention  = nn.MultiheadAttention(
            embed_dim=num_channels[-1], num_heads=num_heads,
            dropout=dropout, batch_first=True)
        self.attn_norm  = nn.LayerNorm(num_channels[-1])
        self.classifier = nn.Sequential(
            nn.Linear(num_channels[-1], 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1))

    def forward(self, x):
        x = self.input_proj(x)
        x = x.transpose(1, 2)
        x = self.tcn(x)
        x = x.transpose(1, 2)
        attn_out, _ = self.attention(x, x, x)
        x = self.attn_norm(x + attn_out)
        x = x.mean(dim=1)
        return self.classifier(x).squeeze(1)


# ── Load models ───────────────────────────────────────────────────────────────
device = torch.device('cpu')

print('Loading XGBoost...')
xgb_model = joblib.load(os.path.join(MODELS_DIR, 'xgboost_model.joblib'))
with open(os.path.join(MODELS_DIR, 'xgboost_features.json')) as f:
    xgb_features = json.load(f)
print('  ✅ XGBoost loaded')

print('Loading TECO...')
teco_model = TECOModel(input_size=23)
teco_model.load_state_dict(torch.load(
    os.path.join(MODELS_DIR, 'teco_best_runpod.pt'),
    map_location=device
))
teco_model.eval()
print('  ✅ TECO loaded')

print('Loading SHAP explainer...')
try:
    xgb_explainer = shap.Explainer(xgb_model)
    print('  ✅ SHAP explainer ready')
except Exception as e:
    print(f'  ⚠ SHAP fallback: {e}')
    xgb_explainer = None

print('Loading AI Agent...')
anthropic_client = anthropic.Anthropic(
    api_key=os.getenv('ANTHROPIC_API_KEY')
)
print('  ✅ AI Agent ready')


# ── Helpers ───────────────────────────────────────────────────────────────────
def build_xgb_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values('hours_from_admission').copy()
    rolling_cols = [
        'heart_rate', 'sbp', 'map', 'resp_rate', 'spo2',
        'creatinine', 'lactate', 'sofa_resp', 'sofa_coag',
        'sofa_liver', 'sofa_cardio', 'sofa_neuro', 'sofa_renal',
    ]
    for col in rolling_cols:
        if col in df.columns:
            df[f'{col}_mean_6h']  = df[col].rolling(6,  min_periods=1).mean()
            df[f'{col}_mean_12h'] = df[col].rolling(12, min_periods=1).mean()
    for col in ['heart_rate', 'map', 'resp_rate', 'spo2',
                'creatinine', 'lactate']:
        if col in df.columns:
            df[f'{col}_change_6h'] = df[col].diff(6).fillna(0)
    for col in ['sofa_resp', 'sofa_coag', 'sofa_liver',
                'sofa_cardio', 'sofa_neuro', 'sofa_renal']:
        if col in df.columns:
            df[f'{col}_max_12h'] = df[col].rolling(12, min_periods=1).max()
    return df.fillna(0)


def build_teco_sequence(df: pd.DataFrame) -> torch.Tensor:
    df = df.sort_values('hours_from_admission')
    X  = df[FEATURE_COLS].values.astype(np.float32)
    if len(X) > MAX_SEQ_LEN:
        X = X[-MAX_SEQ_LEN:]
    elif len(X) < MAX_SEQ_LEN:
        pad = np.zeros((MAX_SEQ_LEN - len(X), len(FEATURE_COLS)),
                       dtype=np.float32)
        X   = np.vstack([pad, X])
    return torch.tensor(X, dtype=torch.float32).unsqueeze(0)


def get_shap_explanation(X_last: pd.DataFrame) -> list:
    try:
        if xgb_explainer is not None:
            shap_vals_obj = xgb_explainer(X_last)
            shap_vals     = shap_vals_obj.values
            if shap_vals.ndim == 3:
                shap_vals = shap_vals[:, :, 1]
            shap_vals = shap_vals[0]
        else:
            shap_vals = xgb_model.feature_importances_

        shap_df = pd.DataFrame({
            'feature' : xgb_features,
            'shap'    : shap_vals,
            'value'   : X_last.values[0],
        }).sort_values('shap', key=abs, ascending=False).head(8)

        return [
            {
                'feature'      : row['feature'],
                'shap_value'   : round(float(row['shap']), 4),
                'actual_value' : round(float(row['value']), 2),
                'direction'    : 'increases risk' if row['shap'] > 0
                                 else 'decreases risk',
            }
            for _, row in shap_df.iterrows()
        ]
    except Exception as e:
        print(f'SHAP failed: {e} — using feature importance fallback')
        importance = xgb_model.feature_importances_
        shap_df = pd.DataFrame({
            'feature' : xgb_features,
            'shap'    : importance,
            'value'   : X_last.values[0],
        }).sort_values('shap', ascending=False).head(8)
        return [
            {
                'feature'      : row['feature'],
                'shap_value'   : round(float(row['shap']), 4),
                'actual_value' : round(float(row['value']), 2),
                'direction'    : 'increases risk',
            }
            for _, row in shap_df.iterrows()
        ]


def generate_clinical_narrative(
    teco_prob  : float,
    xgb_prob   : float,
    risk       : str,
    explanation: list,
    timeline   : dict,
    hours      : int,
) -> dict:

    def last_val(key):
        vals = timeline.get(key, [])
        return round(float(vals[-1]), 1) if vals else 'N/A'

    vitals_summary = f"""
    Heart Rate  : {last_val('heart_rate')} bpm
    MAP         : {last_val('map')} mmHg
    SpO2        : {last_val('spo2')}%
    Resp Rate   : {last_val('resp_rate')}/min
    Lactate     : {last_val('lactate')} mmol/L
    Creatinine  : {last_val('creatinine')} mg/dL
    SOFA Neuro  : {last_val('sofa_neuro')}
    SOFA Cardio : {last_val('sofa_cardio')}
    SOFA Resp   : {last_val('sofa_resp')}
    """

    shap_summary = '\n'.join([
        f"  {i+1}. {e['feature'].replace('_',' ').title()} "
        f"→ {e['direction']} "
        f"(current value: {e['actual_value']})"
        for i, e in enumerate(explanation[:5])
    ])

    prompt = f"""You are a senior ICU physician reviewing a sepsis risk 
assessment for a patient. Provide a concise, direct clinical interpretation.
Do not mention AI, machine learning, models, or algorithms.
Speak as a clinician reviewing the data directly.

RISK ASSESSMENT:
- Primary Risk Score : {teco_prob * 100:.1f}%
- Confirmation Score : {xgb_prob * 100:.1f}%
- Risk Level         : {risk}
- Hours of ICU data  : {hours} hours

LATEST VITALS:
{vitals_summary}

KEY CLINICAL DRIVERS (ranked by importance):
{shap_summary}

Respond ONLY in the following JSON format with no additional text:
{{
  "summary": "3-4 sentence plain English clinical summary of the patient condition and trajectory",
  "urgent_concern": "The single most urgent clinical concern in one sentence",
  "time_to_treatment": "IMMEDIATE (within 30 min) | URGENT (within 1 hour) | SOON (within 2 hours) | ROUTINE (4 hour reassessment)",
  "time_reasoning": "One sentence explaining why this urgency level",
  "next_steps": [
    "Specific action 1",
    "Specific action 2",
    "Specific action 3"
  ],
  "monitor_closely": "What to watch in the next hour and why"
}}"""

    try:
        response = anthropic_client.messages.create(
            model      = 'claude-opus-4-6',
            max_tokens = 1000,
            messages   = [{'role': 'user', 'content': prompt}]
        )
        raw  = response.content[0].text.strip()
        raw  = raw.replace('```json', '').replace('```', '').strip()
        return json.loads(raw)

    except Exception as e:
        print(f'AI Agent error: {e}')
        return {
            'summary'          : 'Clinical narrative unavailable. '
                                 'Please review vitals and risk scores directly.',
            'urgent_concern'   : f'Patient flagged as {risk} risk — '
                                 'manual clinical review required.',
            'time_to_treatment': (
                'IMMEDIATE (within 30 min)' if risk == 'HIGH'   else
                'URGENT (within 1 hour)'    if risk == 'MEDIUM' else
                'ROUTINE (4 hour reassessment)'
            ),
            'time_reasoning'   : 'Based on model risk score.',
            'next_steps'       : [
                'Review full vitals trend',
                'Assess clinical status at bedside',
                'Consult senior clinician',
            ],
            'monitor_closely'  : 'All vital signs and laboratory values.',
        }


def risk_level(prob: float) -> str:
    if prob >= 0.7: return 'HIGH'
    if prob >= 0.4: return 'MEDIUM'
    return 'LOW'


def clinical_recommendations(prob: float) -> list:
    if prob >= 0.7:
        return [
            '🔴 Initiate sepsis bundle protocol immediately',
            '🔴 Obtain blood cultures × 2 before antibiotics',
            '🔴 Administer broad-spectrum antibiotics within 1 hour',
            '🔴 IV fluid resuscitation — 30 mL/kg crystalloid',
            '🔴 Measure lactate — repeat if initial > 2 mmol/L',
            '🔴 Reassess haemodynamic status in 1 hour',
        ]
    elif prob >= 0.4:
        return [
            '🟡 Increase monitoring frequency to every 2 hours',
            '🟡 Review and trend lactate levels',
            '🟡 Assess fluid status and urine output',
            '🟡 Consider infectious source workup',
            '🟡 Reassess in 2 hours or sooner if deteriorating',
        ]
    return [
        '🟢 Continue routine monitoring',
        '🟢 Reassess every 4 hours per standard protocol',
    ]


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get('/')
def root():
    return {
        'app'    : 'VitalWatch',
        'version': '1.0.0',
        'status' : 'running',
        'models' : ['TECO', 'XGBoost', 'AI Agent'],
    }


@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    try:
        content = (await file.read()).decode('utf-8', errors='ignore')
        df, fmt = parse_file(file.filename, content)

        if len(df) == 0:
            raise HTTPException(status_code=400,
                                detail='No valid data rows found in file')

        # ── TECO ─────────────────────────────────────────────────────────────
        seq = build_teco_sequence(df)
        with torch.no_grad():
            logit     = teco_model(seq)
            teco_prob = float(torch.sigmoid(logit).item())

        # ── XGBoost + SHAP ────────────────────────────────────────────────────
        df_fe = build_xgb_features(df)
        for col in xgb_features:
            if col not in df_fe.columns:
                df_fe[col] = 0
        X_last   = df_fe[xgb_features].fillna(0).iloc[[-1]]
        xgb_prob = float(xgb_model.predict_proba(X_last)[0][1])

        explanation = get_shap_explanation(X_last)

        # ── Timeline ──────────────────────────────────────────────────────────
        timeline_cols = [
            'hours_from_admission', 'heart_rate', 'sbp', 'map',
            'resp_rate', 'spo2', 'lactate', 'creatinine',
            'sofa_neuro', 'sofa_cardio', 'sofa_resp', 'sofa_renal',
        ]
        timeline_cols = [c for c in timeline_cols if c in df.columns]
        timeline      = df[timeline_cols].tail(48).to_dict(orient='list')

        # ── AI Narrative ──────────────────────────────────────────────────────
        narrative = generate_clinical_narrative(
            teco_prob   = teco_prob,
            xgb_prob    = xgb_prob,
            risk        = risk_level(teco_prob),
            explanation = explanation,
            timeline    = timeline,
            hours       = len(df),
        )

        return {
            'format_detected' : fmt,
            'hours_of_data'   : len(df),
            'teco'            : {
                'probability' : round(teco_prob, 4),
                'risk_percent': round(teco_prob * 100, 1),
                'risk_level'  : risk_level(teco_prob),
                'prediction'  : int(teco_prob >= 0.5),
            },
            'xgboost'         : {
                'probability' : round(xgb_prob, 4),
                'risk_percent': round(xgb_prob * 100, 1),
                'risk_level'  : risk_level(xgb_prob),
                'prediction'  : int(xgb_prob >= 0.5),
                'confirms_teco': (
                    risk_level(teco_prob) == risk_level(xgb_prob)
                ),
            },
            'explanation'     : explanation,
            'timeline'        : timeline,
            'narrative'       : narrative,
            'recommendations' : clinical_recommendations(teco_prob),
            'disclaimer'      : (
                'VitalWatch is a clinical decision support tool. '
                'All predictions must be reviewed by a qualified clinician. '
                'Final treatment decisions rest with the treating physician.'
            ),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))