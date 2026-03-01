# ── streamlit_app.py — VitalWatch Clinical Interface ─────────────────────────
import streamlit as st
import requests
import plotly.graph_objects as go
import time

st.set_page_config(
    page_title = 'VitalWatch — Early Sepsis Detection',
    page_icon  = '🏥',
    layout     = 'wide',
)

API_URL = 'http://localhost:8000'

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    .main { background-color: #F8F9FA; }

    .risk-high   { border-left:5px solid #C0392B; background:#FDF2F2;
                   border-radius:8px; padding:20px 24px; margin-bottom:12px; }
    .risk-medium { border-left:5px solid #E67E22; background:#FDF6EC;
                   border-radius:8px; padding:20px 24px; margin-bottom:12px; }
    .risk-low    { border-left:5px solid #27AE60; background:#F2FAF5;
                   border-radius:8px; padding:20px 24px; margin-bottom:12px; }

    .score-number { font-size:56px; font-weight:700; line-height:1; }
    .score-label  { font-size:14px; font-weight:600; letter-spacing:0.1em;
                    text-transform:uppercase; margin-top:4px; }

    .narrative-box {
        background: white; border:1px solid #E8EAED; border-radius:10px;
        padding:24px 28px; margin:12px 0; line-height:1.8;
    }
    .narrative-label {
        font-size:11px; font-weight:600; letter-spacing:0.12em;
        text-transform:uppercase; color:#9AA0A6; margin-bottom:8px;
    }
    .narrative-text { font-size:15px; color:#1A1A2E; line-height:1.8; }

    .urgency-immediate { background:#C0392B; color:white; border-radius:6px;
                         padding:12px 18px; font-weight:600; font-size:14px;
                         display:inline-block; margin:8px 0; }
    .urgency-urgent    { background:#E67E22; color:white; border-radius:6px;
                         padding:12px 18px; font-weight:600; font-size:14px;
                         display:inline-block; margin:8px 0; }
    .urgency-soon      { background:#F39C12; color:white; border-radius:6px;
                         padding:12px 18px; font-weight:600; font-size:14px;
                         display:inline-block; margin:8px 0; }
    .urgency-routine   { background:#27AE60; color:white; border-radius:6px;
                         padding:12px 18px; font-weight:600; font-size:14px;
                         display:inline-block; margin:8px 0; }

    .step-box {
        background:white; border:1px solid #E8EAED; border-radius:8px;
        padding:14px 18px; margin:6px 0; font-size:14px;
        color:#1A1A2E; line-height:1.5;
    }
    .monitor-box {
        background:white; border:2px solid #F39C12;
        border-radius:8px; padding:16px 20px;
        font-size:14px; color:#1A1A2E;
        margin-top:8px; line-height:1.7;
    }
    .section-label {
        font-size:11px; font-weight:600; letter-spacing:0.12em;
        text-transform:uppercase; color:#9AA0A6; margin:24px 0 10px 0;
    }
    .confirm-ok   { background:#EAF7EF; color:#1E8449; border:1px solid #A9DFBF;
                    border-radius:20px; padding:4px 14px; font-size:12px;
                    font-weight:500; display:inline-block; }
    .confirm-warn { background:#FEF9E7; color:#9A7D0A; border:1px solid #F9E79F;
                    border-radius:20px; padding:4px 14px; font-size:12px;
                    font-weight:500; display:inline-block; }
    .shap-row  { margin-bottom:12px; }
    .shap-feat { font-size:13px; font-weight:500; color:#1A1A2E; margin-bottom:3px; }
    .shap-bar-bg { background:#F1F3F4; border-radius:4px; height:7px; }
    .disclaimer {
        background:#F8F9FA; border:1px solid #E8EAED; border-radius:8px;
        padding:16px 20px; font-size:11px; color:#9AA0A6;
        margin-top:32px; line-height:1.7;
    }
    .loading-stage {
        background:white; border:1px solid #E8EAED; border-radius:8px;
        padding:16px 20px; margin:8px 0; font-size:14px; color:#374151;
    }
    .typewriter { font-size:15px; color:#1A1A2E; line-height:1.8; }
</style>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;gap:14px;margin-bottom:4px;">
    <span style="font-size:28px;">🏥</span>
    <div>
        <div style="font-size:22px;font-weight:700;color:#1A1A2E;">VitalWatch</div>
        <div style="font-size:13px;color:#9AA0A6;margin-top:-2px;">
            Early Sepsis Detection · Clinical Decision Support
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Upload ────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Patient Data Upload</div>',
            unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    'Upload patient vitals file',
    type             = ['csv', 'json', 'hl7', 'txt'],
    label_visibility = 'collapsed'
)

if uploaded_file is None:
    st.markdown("""
    <div style="background:white;border:2px dashed #E8EAED;border-radius:12px;
                padding:48px;text-align:center;margin-top:16px;">
        <div style="font-size:36px;margin-bottom:12px;">📂</div>
        <div style="font-size:15px;font-weight:600;color:#1A1A2E;">
            Upload a patient data file to begin analysis
        </div>
        <div style="font-size:13px;color:#9AA0A6;margin-top:6px;">
            Supported: CSV · JSON · HL7 v2 · FHIR R4
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Multi-stage loading ───────────────────────────────────────────────────────
stage_placeholder = st.empty()

def show_stage(msg, done=False):
    icon = '✅' if done else '⏳'
    stage_placeholder.markdown(f"""
    <div class="loading-stage">{icon} {msg}</div>
    """, unsafe_allow_html=True)

show_stage('Parsing patient file...')
time.sleep(0.3)
show_stage('Running TECO temporal model...')

try:
    response = requests.post(
        f'{API_URL}/predict',
        files   = {'file': (uploaded_file.name,
                            uploaded_file.getvalue(),
                            'application/octet-stream')},
        timeout = 60
    )
    if response.status_code != 200:
        stage_placeholder.empty()
        st.error(f'API Error: {response.json().get("detail", "Unknown")}')
        st.stop()
    result = response.json()
except requests.exceptions.ConnectionError:
    stage_placeholder.empty()
    st.error('Cannot connect to API. Run: uvicorn main:app --reload --port 8000')
    st.stop()

show_stage('Generating XGBoost confirmation + SHAP explanation...')
time.sleep(0.3)
show_stage('AI is summarising clinical findings...', done=False)
time.sleep(0.5)
stage_placeholder.empty()

# ── Unpack ────────────────────────────────────────────────────────────────────
teco      = result['teco']
xgb       = result['xgboost']
explain   = result['explanation']
timeline  = result['timeline']
narrative = result.get('narrative', {})
risk      = teco['risk_level']
prob      = teco['risk_percent']

RISK_META = {
    'HIGH'  : ('#C0392B', 'risk-high',   '🔴', 0.85),
    'MEDIUM': ('#E67E22', 'risk-medium', '🟡', 0.55),
    'LOW'   : ('#27AE60', 'risk-low',    '🟢', 0.20),
}
color, card_cls, emoji, _ = RISK_META[risk]

# ── Animated Gauge + Risk header ──────────────────────────────────────────────
st.markdown('<div class="section-label">Sepsis Risk Assessment</div>',
            unsafe_allow_html=True)

gauge_col, header_col = st.columns([1, 2], gap='large')

with gauge_col:
    gauge_score = prob / 100

    # Colour based on score
    if gauge_score >= 0.7:
        needle_color = '#C0392B'
        bar_colors   = ['#27AE60', '#F39C12', '#E67E22', '#C0392B']
    elif gauge_score >= 0.4:
        needle_color = '#E67E22'
        bar_colors   = ['#27AE60', '#F39C12', '#E67E22', '#C0392B']
    else:
        needle_color = '#27AE60'
        bar_colors   = ['#27AE60', '#F39C12', '#E67E22', '#C0392B']

    fig_gauge = go.Figure(go.Indicator(
        mode  = 'gauge+number',
        value = prob,
        number = dict(
            suffix   = '%',
            font     = dict(size=36, color=color, family='IBM Plex Sans'),
        ),
        gauge = dict(
            axis = dict(
                range    = [0, 100],
                tickwidth = 1,
                tickcolor = '#374151',
                tickfont  = dict(size=11, color='#374151',
                                 family='IBM Plex Sans'),
                tickvals  = [0, 25, 50, 75, 100],
                ticktext  = ['0', '25', '50', '75', '100'],
            ),
            bar  = dict(color=color, thickness=0.25),
            bgcolor     = 'white',
            borderwidth = 0,
            steps = [
                dict(range=[0,  40],  color='#D5F5E3'),
                dict(range=[40, 70],  color='#FDEBD0'),
                dict(range=[70, 100], color='#FADBD8'),
            ],
            threshold = dict(
                line  = dict(color=color, width=4),
                thickness = 0.8,
                value = prob,
            ),
        ),
        title = dict(
            text     = f'<b>{risk} RISK</b>',
            font     = dict(size=13, color=color, family='IBM Plex Sans'),
        ),
        domain = dict(x=[0, 1], y=[0, 1]),
    ))

    fig_gauge.update_layout(
        height        = 250,
        margin        = dict(l=20, r=20, t=40, b=10),
        paper_bgcolor = 'white',
        font          = dict(family='IBM Plex Sans'),
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

with header_col:
    st.markdown(f"""
    <div class="{card_cls}" style="margin-top:8px;">
        <div style="display:flex;align-items:center;gap:32px;">
            <div>
                <div class="score-number" style="color:{color};">
                    {prob:.0f}%
                </div>
                <div class="score-label" style="color:{color};">
                    {emoji} {risk} SEPSIS RISK
                </div>
            </div>
            <div style="flex:1;">
                <div style="background:#E8EAED;border-radius:8px;
                            height:10px;overflow:hidden;margin-bottom:10px;">
                    <div style="background:{color};width:{prob}%;
                                height:100%;border-radius:8px;"></div>
                </div>
                <div style="font-size:12px;color:#6B7280;">
                    TECO primary model &nbsp;·&nbsp;
                    {result['hours_of_data']} hours of data &nbsp;·&nbsp;
                    {result['format_detected'].upper()} format
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    confirms   = xgb['confirms_teco']
    badge_cls  = 'confirm-ok' if confirms else 'confirm-warn'
    badge_text = (
        f'✓ XGBoost confirms — {xgb["risk_percent"]:.0f}% risk'
        if confirms else
        f'⚠ XGBoost diverges — {xgb["risk_percent"]:.0f}% risk · review carefully'
    )
    st.markdown(f'<span class="{badge_cls}">{badge_text}</span>',
                unsafe_allow_html=True)

    # Time to treatment badge
    tti = narrative.get('time_to_treatment', '')
    if tti:
        tti_lower = tti.lower()
        if 'immediate' in tti_lower: tti_cls = 'urgency-immediate'
        elif 'urgent'  in tti_lower: tti_cls = 'urgency-urgent'
        elif 'soon'    in tti_lower: tti_cls = 'urgency-soon'
        else:                        tti_cls = 'urgency-routine'
        st.markdown(f"""
        <div style="margin-top:14px;">
            <div class="narrative-label">Time to Treatment</div>
            <div class="{tti_cls}">⏱ {tti}</div>
            <div style="font-size:12px;color:#9AA0A6;margin-top:6px;">
                {narrative.get('time_reasoning','')}
            </div>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ── AI Narrative with typewriter effect ───────────────────────────────────────
st.markdown('<div class="section-label">AI Clinical Summary</div>',
            unsafe_allow_html=True)

summary = narrative.get('summary', '')
concern = narrative.get('urgent_concern', '')

if summary:
    st.markdown(
        '<div class="narrative-label">Clinical Assessment</div>',
        unsafe_allow_html=True
    )

    summary_placeholder = st.empty()
    displayed = ''

    for char in summary:
        displayed += char
        summary_placeholder.markdown(f"""
        <div class="narrative-box">
            <div class="typewriter">{displayed}▌</div>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(0.012)

    # Final render without cursor
    summary_placeholder.markdown(f"""
    <div class="narrative-box">
        <div class="typewriter">{displayed}</div>
    </div>
    """, unsafe_allow_html=True)

if concern:
    st.markdown(f"""
    <div class="narrative-box" style="border-left:4px solid {color};">
        <div class="narrative-label">Most Urgent Concern</div>
        <div class="narrative-text" style="font-weight:500;">{concern}</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ── Two column layout ─────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 1], gap='large')

with left_col:

    # ── Next steps ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Recommended Next Steps</div>',
                unsafe_allow_html=True)

    steps = narrative.get('next_steps', [])
    for i, step in enumerate(steps, 1):
        st.markdown(f"""
        <div class="step-box">
            <span style="background:#1A1A2E;color:white;border-radius:50%;
                         width:22px;height:22px;display:inline-flex;
                         align-items:center;justify-content:center;
                         font-size:11px;font-weight:600;margin-right:12px;">
                {i}
            </span>
            {step}
        </div>
        """, unsafe_allow_html=True)

    # ── Monitor Closely — FIXED ───────────────────────────────────────────────
    monitor = narrative.get('monitor_closely', '')
    if monitor:
        st.markdown('<div class="section-label">Monitor Closely</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div class="monitor-box">
            <span style="font-size:16px;">👁</span>
            <span style="margin-left:10px;font-size:14px;
                         color:#1A1A2E;font-weight:400;">
                {monitor}
            </span>
        </div>
        """, unsafe_allow_html=True)

    # ── SHAP ─────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Key Risk Drivers (SHAP)</div>',
                unsafe_allow_html=True)

    max_shap = max(abs(e['shap_value']) for e in explain) if explain else 1
    for item in explain:
        sv      = item['shap_value']
        val     = item['actual_value']
        feat    = item['feature'].replace('_', ' ').title()
        pct     = abs(sv) / max_shap * 100
        bar_col = '#C0392B' if sv > 0 else '#27AE60'
        arrow   = '↑' if sv > 0 else '↓'

        st.markdown(f"""
        <div class="shap-row">
            <div style="display:flex;justify-content:space-between;
                        margin-bottom:4px;">
                <span class="shap-feat">{feat}</span>
                <span style="font-size:12px;color:{bar_col};font-weight:500;">
                    {arrow} &nbsp;{val}
                </span>
            </div>
            <div class="shap-bar-bg">
                <div style="background:{bar_col};width:{pct:.0f}%;
                            height:7px;border-radius:4px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

with right_col:

    st.markdown('<div class="section-label">Vitals Timeline</div>',
                unsafe_allow_html=True)

    hours_axis = timeline.get(
        'hours_from_admission',
        list(range(len(next(iter(timeline.values()), []))))
    )

    # ── Shared chart style ────────────────────────────────────────────────────
    CHART_STYLE = dict(
        paper_bgcolor = 'white',
        plot_bgcolor  = '#FAFAFA',
        font          = dict(family='IBM Plex Sans', size=12, color='#374151'),
        margin        = dict(l=10, r=10, t=40, b=60),
        height        = 230,
        legend        = dict(
            orientation = 'h',
            y           = -0.25,
            x           = 0,
            font        = dict(size=11, color='#374151'),
        ),
        xaxis = dict(
            title      = dict(text='Hours from admission',
                              font=dict(size=11, color='#374151')),
            showgrid   = True,
            gridcolor  = '#E8EAED',
            gridwidth  = 1,
            tickfont   = dict(size=10, color='#374151'),
            zeroline   = False,
        ),
        yaxis = dict(
            showgrid  = True,
            gridcolor = '#E8EAED',
            gridwidth = 1,
            tickfont  = dict(size=10, color='#374151'),
            zeroline  = False,
        ),
    )

    # ── Chart 1 — Haemodynamics ───────────────────────────────────────────────
    fig1 = go.Figure()
    if 'heart_rate' in timeline:
        fig1.add_trace(go.Scatter(
            x=hours_axis, y=timeline['heart_rate'],
            name='Heart Rate (bpm)',
            line=dict(color='#C0392B', width=2),
            marker=dict(size=4)
        ))
    if 'map' in timeline:
        fig1.add_trace(go.Scatter(
            x=hours_axis, y=timeline['map'],
            name='MAP (mmHg)',
            line=dict(color='#2980B9', width=2),
            marker=dict(size=4)
        ))
        # Threshold as separate trace (cleaner than hline annotation)
        fig1.add_trace(go.Scatter(
            x=hours_axis,
            y=[65] * len(hours_axis),
            name='MAP threshold (65)',
            line=dict(color='#2980B9', width=1, dash='dot'),
            mode='lines'
        ))
    if 'resp_rate' in timeline:
        fig1.add_trace(go.Scatter(
            x=hours_axis, y=timeline['resp_rate'],
            name='Resp Rate (/min)',
            line=dict(color='#27AE60', width=2),
            marker=dict(size=4)
        ))

    fig1.update_layout(
        title=dict(text='Haemodynamics',
                   font=dict(size=13, color='#1A1A2E', family='IBM Plex Sans')),
        **CHART_STYLE
    )
    st.plotly_chart(fig1, use_container_width=True)

    # ── Chart 2 — SpO2 + Lactate (separate Y axes, clean) ────────────────────
    fig2 = go.Figure()
    if 'spo2' in timeline:
        fig2.add_trace(go.Scatter(
            x=hours_axis, y=timeline['spo2'],
            name='SpO2 (%)',
            line=dict(color='#8E44AD', width=2),
            marker=dict(size=4),
            yaxis='y1'
        ))
        fig2.add_trace(go.Scatter(
            x=hours_axis,
            y=[95] * len(hours_axis),
            name='SpO2 threshold (95%)',
            line=dict(color='#8E44AD', width=1, dash='dot'),
            mode='lines',
            yaxis='y1'
        ))
    if 'lactate' in timeline:
        fig2.add_trace(go.Scatter(
            x=hours_axis, y=timeline['lactate'],
            name='Lactate (mmol/L)',
            line=dict(color='#E67E22', width=2),
            marker=dict(size=4),
            yaxis='y2'
        ))
        fig2.add_trace(go.Scatter(
            x=hours_axis,
            y=[2.0] * len(hours_axis),
            name='Lactate threshold (2.0)',
            line=dict(color='#E67E22', width=1, dash='dot'),
            mode='lines',
            yaxis='y2'
        ))

    fig2.update_layout(
        title=dict(text='Oxygenation & Lactate',
                   font=dict(size=13, color='#1A1A2E', family='IBM Plex Sans')),
        yaxis  = dict(
            title    = dict(text='SpO2 (%)',
                            font=dict(size=11, color='#8E44AD')),
            tickfont = dict(size=10, color='#8E44AD'),
            showgrid = True, gridcolor='#E8EAED',
            range    = [80, 102],
            zeroline = False,
        ),
        yaxis2 = dict(
            title     = dict(text='Lactate (mmol/L)',
                             font=dict(size=11, color='#E67E22')),
            tickfont  = dict(size=10, color='#E67E22'),
            overlaying = 'y',
            side       = 'right',
            showgrid   = False,
            range      = [0, 10],
            zeroline   = False,
        ),
        **{k: v for k, v in CHART_STYLE.items() if k != 'yaxis'}
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── Chart 3 — SOFA ────────────────────────────────────────────────────────
    fig3  = go.Figure()
    sofas = {
        'sofa_neuro'  : ('SOFA Neuro',  '#1A1A2E'),
        'sofa_cardio' : ('SOFA Cardio', '#C0392B'),
        'sofa_resp'   : ('SOFA Resp',   '#2980B9'),
        'sofa_renal'  : ('SOFA Renal',  '#27AE60'),
    }
    for col, (label, lcolor) in sofas.items():
        if col in timeline:
            fig3.add_trace(go.Scatter(
                x=hours_axis, y=timeline[col],
                name=label,
                line=dict(color=lcolor, width=2),
                marker=dict(size=4)
            ))

    fig3.update_layout(
        title=dict(text='SOFA Components',
                   font=dict(size=13, color='#1A1A2E', family='IBM Plex Sans')),
        yaxis=dict(
            title    = dict(text='Score (0–4)',
                            font=dict(size=11, color='#374151')),
            tickfont = dict(size=10, color='#374151'),
            showgrid = True, gridcolor='#E8EAED',
            range    = [0, 4.5],
            zeroline = False,
        ),
        **{k: v for k, v in CHART_STYLE.items() if k != 'yaxis'}
    )
    st.plotly_chart(fig3, use_container_width=True)

# ── Disclaimer ────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="disclaimer">
    <strong>Clinical Decision Support Disclaimer</strong><br>
    {result['disclaimer']}<br><br>
    <strong>Regulatory:</strong> VitalWatch is a software-based clinical
    decision support tool. It does not replace clinical judgment and is
    not FDA-cleared for autonomous clinical decision making.
    All outputs must be reviewed by a qualified healthcare professional.
</div>
""", unsafe_allow_html=True)