# ── parser.py — Universal Clinical File Parser ────────────────────────────────
import pandas as pd
import numpy as np
import json
import io
from typing import Tuple

# ── Standard feature columns ──────────────────────────────────────────────────
FEATURE_COLS = [
    'heart_rate', 'sbp', 'dbp', 'map',
    'resp_rate', 'temperature', 'spo2',
    'creatinine', 'platelets', 'bilirubin',
    'lactate', 'hemoglobin', 'wbc',
    'glucose', 'potassium', 'sodium',
    'sofa_resp', 'sofa_coag', 'sofa_liver',
    'sofa_cardio', 'sofa_neuro', 'sofa_renal',
    'hours_from_admission',
]

# ── All possible aliases → normalised name ────────────────────────────────────
# Every possible column name a JSON/CSV/HL7 file might use
ALIASES = {
    # Heart rate
    'heart_rate'              : 'heart_rate',
    'heartrate'               : 'heart_rate',
    'hr'                      : 'heart_rate',
    'pulse'                   : 'heart_rate',
    'pulse_rate'              : 'heart_rate',
    'heart rate'              : 'heart_rate',

    # Blood pressure
    'sbp'                     : 'sbp',
    'systolic'                : 'sbp',
    'systolic_bp'             : 'sbp',
    'systolicbp'              : 'sbp',
    'systolic_blood_pressure' : 'sbp',
    'sys_bp'                  : 'sbp',
    'dbp'                     : 'dbp',
    'diastolic'               : 'dbp',
    'diastolic_bp'            : 'dbp',
    'diastolicbp'             : 'dbp',
    'diastolic_blood_pressure': 'dbp',
    'dia_bp'                  : 'dbp',
    'map'                     : 'map',
    'mean_arterial_pressure'  : 'map',
    'mean arterial pressure'  : 'map',
    'meanarterialpressure'    : 'map',

    # Respiratory
    'resp_rate'               : 'resp_rate',
    'respiratory_rate'        : 'resp_rate',
    'respiratoryrate'         : 'resp_rate',
    'rr'                      : 'resp_rate',
    'breaths_per_min'         : 'resp_rate',
    'respiration_rate'        : 'resp_rate',

    # Temperature
    'temperature'             : 'temperature',
    'temp'                    : 'temperature',
    'body_temp'               : 'temperature',
    'body_temperature'        : 'temperature',
    'temp_c'                  : 'temperature',
    'temperature_c'           : 'temperature',
    'temp_f'                  : 'temperature',

    # SpO2
    'spo2'                    : 'spo2',
    'o2sat'                   : 'spo2',
    'o2_sat'                  : 'spo2',
    'oxygen_saturation'       : 'spo2',
    'oxygensaturation'        : 'spo2',
    'spo2_pct'                : 'spo2',
    'pulse_ox'                : 'spo2',
    'pulseox'                 : 'spo2',
    'sat'                     : 'spo2',

    # Labs
    'creatinine'              : 'creatinine',
    'cr'                      : 'creatinine',
    'creat'                   : 'creatinine',
    'serum_creatinine'        : 'creatinine',

    'platelets'               : 'platelets',
    'plt'                     : 'platelets',
    'platelet_count'          : 'platelets',
    'platelet count'          : 'platelets',

    'bilirubin'               : 'bilirubin',
    'bili'                    : 'bilirubin',
    'total_bilirubin'         : 'bilirubin',
    'tbili'                   : 'bilirubin',
    't_bili'                  : 'bilirubin',

    'lactate'                 : 'lactate',
    'lac'                     : 'lactate',
    'lactic_acid'             : 'lactate',
    'lacticacid'              : 'lactate',
    'blood_lactate'           : 'lactate',

    'hemoglobin'              : 'hemoglobin',
    'haemoglobin'             : 'hemoglobin',
    'hgb'                     : 'hemoglobin',
    'hb'                      : 'hemoglobin',
    'hbg'                     : 'hemoglobin',

    'wbc'                     : 'wbc',
    'wbc_count'               : 'wbc',
    'white_blood_cells'       : 'wbc',
    'white_blood_count'       : 'wbc',
    'leucocytes'              : 'wbc',
    'leukocytes'              : 'wbc',
    'wcc'                     : 'wbc',

    'glucose'                 : 'glucose',
    'blood_glucose'           : 'glucose',
    'glu'                     : 'glucose',
    'gluc'                    : 'glucose',
    'blood_sugar'             : 'glucose',

    'potassium'               : 'potassium',
    'k'                       : 'potassium',
    'serum_potassium'         : 'potassium',

    'sodium'                  : 'sodium',
    'na'                      : 'sodium',
    'serum_sodium'            : 'sodium',

    # GCS (needed for SOFA Neuro computation)
    'gcs'                     : 'gcs',
    'gcs_total'               : 'gcs',
    'glasgow_coma_scale'      : 'gcs',
    'glasgow_coma_score'      : 'gcs',
    'gcs_score'               : 'gcs',

    # PaO2 / FiO2 (needed for SOFA Resp computation)
    'pao2'                    : 'pao2',
    'pa_o2'                   : 'pao2',
    'arterial_po2'            : 'pao2',
    'po2'                     : 'pao2',

    'fio2'                    : 'fio2',
    'fi_o2'                   : 'fio2',
    'fraction_inspired_o2'    : 'fio2',
    'inspired_o2'             : 'fio2',

    # Vasopressor flags / doses (for SOFA Cardio)
    'vasopressor'             : 'vasopressor',
    'vasopressor_dose'        : 'vasopressor',
    'norepinephrine'          : 'norepinephrine',
    'norepinephrine_dose'     : 'norepinephrine',
    'norepi'                  : 'norepinephrine',
    'dopamine'                : 'dopamine',
    'dopamine_dose'           : 'dopamine',
    'dobutamine'              : 'dobutamine',
    'dobutamine_dose'         : 'dobutamine',
    'epinephrine'             : 'epinephrine',
    'epinephrine_dose'        : 'epinephrine',

    # Urine output (for SOFA Renal)
    'urine_output'            : 'urine_output',
    'urine_output_ml'         : 'urine_output_hourly',
    'urine_output_ml_hr'      : 'urine_output_hourly',
    'uo_ml_hr'                : 'urine_output_hourly',
    'uo_ml'                   : 'urine_output_hourly',
    'urine_output_24h'        : 'urine_output',
    'uo_24h'                  : 'urine_output',
    'urine_24h'               : 'urine_output',

    # SOFA (pre-computed — keep if present)
    'sofa_resp'               : 'sofa_resp',
    'sofa_respiratory'        : 'sofa_resp',
    'sofa_coag'               : 'sofa_coag',
    'sofa_coagulation'        : 'sofa_coag',
    'sofa_liver'              : 'sofa_liver',
    'sofa_hepatic'            : 'sofa_liver',
    'sofa_cardio'             : 'sofa_cardio',
    'sofa_cardiovascular'     : 'sofa_cardio',
    'sofa_cardiovasc'         : 'sofa_cardio',
    'sofa_neuro'              : 'sofa_neuro',
    'sofa_neurological'       : 'sofa_neuro',
    'sofa_cns'                : 'sofa_neuro',
    'sofa_renal'              : 'sofa_renal',
    'sofa_kidney'             : 'sofa_renal',

    # Time
    'hours_from_admission'    : 'hours_from_admission',
    'hours'                   : 'hours_from_admission',
    'hour'                    : 'hours_from_admission',
    'time_hours'              : 'hours_from_admission',
    'admission_hour'          : 'hours_from_admission',
    'icu_hour'                : 'hours_from_admission',
    'time'                    : 'hours_from_admission',
    'timestamp'               : 'hours_from_admission',
}

# ── FHIR LOINC → feature ──────────────────────────────────────────────────────
LOINC_MAP = {
    '8867-4'  : 'heart_rate',
    '8480-6'  : 'sbp',
    '8462-4'  : 'dbp',
    '8478-0'  : 'map',
    '9279-1'  : 'resp_rate',
    '8310-5'  : 'temperature',
    '2708-6'  : 'spo2',
    '59408-5' : 'spo2',
    '38483-4' : 'creatinine',
    '777-3'   : 'platelets',
    '1975-2'  : 'bilirubin',
    '2524-7'  : 'lactate',
    '718-7'   : 'hemoglobin',
    '6690-2'  : 'wbc',
    '2345-7'  : 'glucose',
    '2823-3'  : 'potassium',
    '2951-2'  : 'sodium',
    '9269-2'  : 'gcs',          # GCS total
    '19994-3' : 'pao2',         # Arterial PaO2
    '3150-0'  : 'fio2',         # FiO2
}

# ── HL7 OBX → feature ─────────────────────────────────────────────────────────
HL7_MAP = {
    'HR': 'heart_rate', 'PULSE': 'heart_rate',
    'SBP': 'sbp', 'DBP': 'dbp', 'MAP': 'map',
    'RR': 'resp_rate', 'RESP': 'resp_rate',
    'TEMP': 'temperature',
    'SPO2': 'spo2', 'O2SAT': 'spo2',
    'CREAT': 'creatinine',
    'PLT': 'platelets',
    'TBILI': 'bilirubin',
    'LACT': 'lactate',
    'HGB': 'hemoglobin', 'HB': 'hemoglobin',
    'WBC': 'wbc',
    'GLUC': 'glucose',
    'K': 'potassium',
    'NA': 'sodium',
    'GCS': 'gcs',
    'PAO2': 'pao2', 'PO2': 'pao2',
    'FIO2': 'fio2',
}

# ── Cohort medians for imputation ─────────────────────────────────────────────
MEDIANS = {
    'heart_rate': 84.0,   'sbp': 117.0,
    'dbp': 62.0,          'map': 81.7,
    'resp_rate': 20.0,    'temperature': 98.4,
    'spo2': 97.0,         'creatinine': 0.9,
    'platelets': 195.0,   'bilirubin': 0.6,
    'lactate': 1.4,       'hemoglobin': 9.7,
    'wbc': 10.8,          'glucose': 125.0,
    'potassium': 4.0,     'sodium': 139.0,
    'sofa_resp': 0.0,     'sofa_coag': 0.0,
    'sofa_liver': 0.0,    'sofa_cardio': 0.0,
    'sofa_neuro': 0.0,    'sofa_renal': 0.0,
    'hours_from_admission': 24.0,
}


# ══════════════════════════════════════════════════════════════════════════════
#  SOFA SCORE COMPUTATION FROM RAW CLINICAL DATA
# ══════════════════════════════════════════════════════════════════════════════

def _sofa_respiratory(row) -> int:
    """
    SOFA Respiratory component.
    Priority: PaO2/FiO2 ratio → SpO2-based estimate → resp_rate heuristic.

    Standard SOFA (PaO2/FiO2):
        ≥ 400       → 0
        300 – 399   → 1
        200 – 299   → 2
        100 – 199   → 3  (with mechanical ventilation)
        < 100       → 4  (with mechanical ventilation)

    SpO2/FiO2 proxy (validated surrogate — Rice et al. Crit Care Med 2007):
        SpO2/FiO2 ≥ 512 → PF ≈ ≥400 → 0
        SpO2/FiO2 357–511 → PF ≈ 300–399 → 1
        SpO2/FiO2 214–356 → PF ≈ 200–299 → 2
        SpO2/FiO2 89–213  → PF ≈ 100–199 → 3
        SpO2/FiO2 < 89    → PF ≈ <100 → 4

    SpO2-only fallback (no FiO2 known, assume room air FiO2=0.21):
        ≥ 97 → 0    (normal oxygenation)
        94–96 → 1   (mild impairment)
        90–93 → 2   (moderate impairment)
        85–89 → 3   (severe impairment)
        < 85 → 4    (critical impairment)

    Last resort — resp_rate heuristic:
        ≤ 20 → 0,  21–25 → 1,  26–30 → 2,  31–35 → 3,  >35 → 4
    """
    pao2 = row.get('pao2', np.nan)
    fio2 = row.get('fio2', np.nan)
    spo2 = row.get('spo2', np.nan)
    rr   = row.get('resp_rate', np.nan)

    # ── Best: PaO2/FiO2 ratio ────────────────────────────────────────────────
    if pd.notna(pao2) and pd.notna(fio2) and fio2 > 0:
        # Normalise FiO2 if given as percentage (e.g. 40 instead of 0.40)
        if fio2 > 1.0:
            fio2 = fio2 / 100.0
        pf = pao2 / fio2
        if pf >= 400: return 0
        if pf >= 300: return 1
        if pf >= 200: return 2
        if pf >= 100: return 3
        return 4

    # ── Good: SpO2/FiO2 proxy ────────────────────────────────────────────────
    if pd.notna(spo2) and pd.notna(fio2) and fio2 > 0:
        if fio2 > 1.0:
            fio2 = fio2 / 100.0
        sf = spo2 / fio2
        if sf >= 512: return 0
        if sf >= 357: return 1
        if sf >= 214: return 2
        if sf >= 89:  return 3
        return 4

    # ── Acceptable: SpO2 alone (assume room air) ─────────────────────────────
    if pd.notna(spo2):
        if spo2 >= 97: return 0
        if spo2 >= 94: return 1
        if spo2 >= 90: return 2
        if spo2 >= 85: return 3
        return 4

    # ── Last resort: respiratory rate heuristic ──────────────────────────────
    if pd.notna(rr):
        if rr <= 20: return 0
        if rr <= 25: return 1
        if rr <= 30: return 2
        if rr <= 35: return 3
        return 4

    return 0  # no data → assume normal


def _sofa_coagulation(row) -> int:
    """
    SOFA Coagulation component — platelet count (×10³/µL).
        ≥ 150   → 0
        100–149 → 1
        50–99   → 2
        20–49   → 3
        < 20    → 4
    """
    plt_val = row.get('platelets', np.nan)
    if pd.isna(plt_val):
        return 0
    if plt_val >= 150: return 0
    if plt_val >= 100: return 1
    if plt_val >= 50:  return 2
    if plt_val >= 20:  return 3
    return 4


def _sofa_liver(row) -> int:
    """
    SOFA Liver component — total bilirubin (mg/dL).
        < 1.2    → 0
        1.2–1.9  → 1
        2.0–5.9  → 2
        6.0–11.9 → 3
        ≥ 12.0   → 4
    """
    bili = row.get('bilirubin', np.nan)
    if pd.isna(bili):
        return 0
    if bili < 1.2:  return 0
    if bili < 2.0:  return 1
    if bili < 6.0:  return 2
    if bili < 12.0: return 3
    return 4


def _sofa_cardiovascular(row) -> int:
    """
    SOFA Cardiovascular component.
    Standard:
        MAP ≥ 70, no vasopressors            → 0
        MAP < 70                              → 1
        Dopamine ≤ 5 or any dobutamine        → 2
        Dopamine > 5 OR norepi/epi ≤ 0.1      → 3
        Dopamine > 15 OR norepi/epi > 0.1      → 4

    When vasopressor data is unavailable, we use MAP + lactate heuristic:
        MAP ≥ 70                              → 0
        MAP 65–69                             → 1
        MAP 55–64 OR (MAP < 70 + lactate > 2) → 2
        MAP < 55 OR (MAP < 65 + lactate > 4)  → 3
        MAP < 50 + lactate > 4                → 4
    """
    map_val = row.get('map', np.nan)
    lac     = row.get('lactate', np.nan)

    # Vasopressor data (µg/kg/min)
    norepi  = row.get('norepinephrine', np.nan)
    dopa    = row.get('dopamine', np.nan)
    dobut   = row.get('dobutamine', np.nan)
    epi     = row.get('epinephrine', np.nan)
    vaso    = row.get('vasopressor', np.nan)

    has_vasopressor = any(
        pd.notna(v) and v > 0
        for v in [norepi, dopa, dobut, epi, vaso]
    )

    # ── With vasopressor data ─────────────────────────────────────────────────
    if has_vasopressor:
        norepi_val = norepi if pd.notna(norepi) else 0
        epi_val    = epi    if pd.notna(epi)    else 0
        dopa_val   = dopa   if pd.notna(dopa)   else 0

        if dopa_val > 15 or norepi_val > 0.1 or epi_val > 0.1:
            return 4
        if dopa_val > 5 or norepi_val > 0 or epi_val > 0:
            return 3
        if pd.notna(dobut) and dobut > 0:
            return 2
        if dopa_val > 0:
            return 2
        return 1  # on vasopressors but low dose

    # ── Without vasopressor data — MAP + lactate heuristic ────────────────────
    if pd.isna(map_val):
        return 0

    lac_val = lac if pd.notna(lac) else 0

    if map_val >= 70:
        return 0
    if map_val >= 65:
        # Mild hypotension — check if lactate suggests worse perfusion
        if lac_val > 2:
            return 2
        return 1
    if map_val >= 55:
        if lac_val > 4:
            return 3
        return 2
    # MAP < 55
    if lac_val > 4:
        return 4
    return 3


def _sofa_neurological(row) -> int:
    """
    SOFA Neurological component — Glasgow Coma Scale.
        GCS 15      → 0
        GCS 13–14   → 1
        GCS 10–12   → 2
        GCS 6–9     → 3
        GCS < 6     → 4

    Without GCS data → 0 (assumed normal / not assessable).
    """
    gcs = row.get('gcs', np.nan)
    if pd.isna(gcs):
        return 0
    gcs = int(gcs)
    if gcs >= 15: return 0
    if gcs >= 13: return 1
    if gcs >= 10: return 2
    if gcs >= 6:  return 3
    return 4


def _sofa_renal(row) -> int:
    """
    SOFA Renal component — creatinine (mg/dL) or urine output.

    Creatinine:
        < 1.2    → 0
        1.2–1.9  → 1
        2.0–3.4  → 2
        3.5–4.9  → 3
        ≥ 5.0    → 4

    Urine output (mL/day, if available):
        ≥ 500    → max(creat_score, 0)
        200–499  → max(creat_score, 3)
        < 200    → max(creat_score, 4)
    """
    creat = row.get('creatinine', np.nan)
    uo    = row.get('urine_output', np.nan)

    creat_score = 0
    if pd.notna(creat):
        if creat < 1.2:    creat_score = 0
        elif creat < 2.0:  creat_score = 1
        elif creat < 3.5:  creat_score = 2
        elif creat < 5.0:  creat_score = 3
        else:               creat_score = 4

    uo_score = 0
    if pd.notna(uo):
        if uo < 200:   uo_score = 4
        elif uo < 500: uo_score = 3

    return max(creat_score, uo_score)


def compute_sofa_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 6 SOFA component scores row-by-row from raw clinical data.

    Only overwrites a SOFA column if:
      (a) it doesn't exist in the DataFrame, OR
      (b) it exists but is ALL zeros (i.e. was median-imputed, not real data)

    This preserves genuinely pre-computed SOFA scores from MIMIC/eICU extracts
    while filling in computed scores for raw vitals-only files.
    """
    df = df.copy()

    sofa_components = {
        'sofa_resp'   : _sofa_respiratory,
        'sofa_coag'   : _sofa_coagulation,
        'sofa_liver'  : _sofa_liver,
        'sofa_cardio' : _sofa_cardiovascular,
        'sofa_neuro'  : _sofa_neurological,
        'sofa_renal'  : _sofa_renal,
    }

    computed_cols = []

    for col, scorer in sofa_components.items():
        # Check if column has real data (not all zeros / not absent)
        has_real_data = (
            col in df.columns
            and not (df[col] == 0).all()
            and not df[col].isna().all()
        )

        if has_real_data:
            print(f'  ✓ {col}: keeping pre-computed values '
                  f'(range {df[col].min():.0f}–{df[col].max():.0f})')
            continue

        # Compute from raw data
        df[col] = df.apply(scorer, axis=1).astype(float)
        computed_cols.append(col)

        score_range = f'{df[col].min():.0f}–{df[col].max():.0f}'
        non_zero    = (df[col] > 0).sum()
        print(f'  ⚕ {col}: COMPUTED from raw data '
              f'(range {score_range}, {non_zero}/{len(df)} non-zero)')

    if computed_cols:
        print(f'  → Computed {len(computed_cols)} SOFA components: '
              f'{computed_cols}')
    else:
        print(f'  → All 6 SOFA components had pre-computed values')

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  COLUMN NORMALISATION
# ══════════════════════════════════════════════════════════════════════════════

def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise all column names using the alias map.
    Handles: spaces, mixed case, underscores, camelCase.
    """
    rename_map = {}
    for col in df.columns:
        # Try exact match first
        key = col.lower().strip().replace(' ', '_').replace('-', '_')
        if key in ALIASES:
            rename_map[col] = ALIASES[key]
        # Try removing all underscores
        elif key.replace('_', '') in {
            k.replace('_', '') for k in ALIASES
        }:
            for alias_key in ALIASES:
                if alias_key.replace('_', '') == key.replace('_', ''):
                    rename_map[col] = ALIASES[alias_key]
                    break

    return df.rename(columns=rename_map)


# ══════════════════════════════════════════════════════════════════════════════
#  FORMAT DETECTION & PARSERS
# ══════════════════════════════════════════════════════════════════════════════

def flatten_json_record(record: dict) -> dict:
    """
    Flatten a potentially nested JSON record into a flat dict.
    Handles common clinical JSON nesting patterns.
    """
    flat = {}

    def _flatten(obj, prefix=''):
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_key = f'{prefix}_{k}' if prefix else k
                _flatten(v, new_key)
        elif isinstance(obj, (int, float, str)):
            # Only keep scalar values
            try:
                flat[prefix] = float(obj)
            except (ValueError, TypeError):
                flat[prefix] = obj
        # Skip lists, nulls

    _flatten(record)
    return flat


def detect_format(filename: str, content: str) -> str:
    """Detect file format."""
    fname = filename.lower()
    if fname.endswith('.csv'):
        return 'csv'
    if fname.endswith('.hl7') or fname.endswith('.txt'):
        if content.strip().startswith('MSH|'):
            return 'hl7'
    if fname.endswith('.json'):
        try:
            data = json.loads(content)
            if isinstance(data, dict) and data.get('resourceType'):
                return 'fhir'
            return 'json'
        except:
            return 'csv'
    # Content sniffing fallback
    stripped = content.strip()
    if stripped.startswith('MSH|'):
        return 'hl7'
    if stripped.startswith('{') or stripped.startswith('['):
        try:
            data = json.loads(content)
            if isinstance(data, dict) and data.get('resourceType'):
                return 'fhir'
            return 'json'
        except:
            pass
    return 'csv'


def parse_csv(content: str) -> pd.DataFrame:
    """Parse CSV → normalise columns."""
    df = pd.read_csv(io.StringIO(content))
    df = normalise_columns(df)

    # Derive MAP if missing
    if 'map' not in df.columns and \
       'sbp' in df.columns and 'dbp' in df.columns:
        df['map'] = (df['sbp'] + 2 * df['dbp']) / 3

    if 'hours_from_admission' not in df.columns:
        df['hours_from_admission'] = range(len(df))

    return df


def parse_json(content: str) -> pd.DataFrame:
    """
    Parse JSON — handles ALL common structures:
    1. List of hourly records  → [{hr:80, sbp:120, ...}, ...]
    2. Dict with data array    → {data: [{...}, ...]}
    3. Dict with parallel arrays → {heart_rate: [80,82,...], sbp: [...]}
    4. Single record dict      → {hr: 80, sbp: 120}
    5. Nested records          → [{vitals: {hr:80}, labs: {lac:1.4}}]
    """
    data = json.loads(content)

    # ── Structure 1: List of records ──────────────────────────────────────────
    if isinstance(data, list):
        if len(data) == 0:
            raise ValueError('JSON array is empty')

        # Flatten each record in case of nesting
        flat_records = [flatten_json_record(r)
                        if isinstance(r, dict) else r
                        for r in data]
        df = pd.DataFrame(flat_records)

    # ── Structure 2: Dict ─────────────────────────────────────────────────────
    elif isinstance(data, dict):

        # Check for a nested data/records/observations array
        array_keys = ['data', 'records', 'observations',
                      'vitals', 'measurements', 'rows', 'entries']
        found_array = None
        for key in array_keys:
            if key in data and isinstance(data[key], list):
                found_array = key
                break

        if found_array:
            # Structure 2a → {data: [{...}, ...]}
            flat_records = [flatten_json_record(r)
                            if isinstance(r, dict) else r
                            for r in data[found_array]]
            df = pd.DataFrame(flat_records)

        else:
            # Check if values are lists (parallel arrays)
            # Structure 2b → {heart_rate: [80,82,...], sbp: [...]}
            list_vals = {k: v for k, v in data.items()
                         if isinstance(v, list)}
            scalar_vals = {k: v for k, v in data.items()
                           if isinstance(v, (int, float))}

            if list_vals:
                df = pd.DataFrame(list_vals)
            elif scalar_vals:
                # Structure 2c → single record
                df = pd.DataFrame([scalar_vals])
            else:
                # Last resort — flatten everything
                flat = flatten_json_record(data)
                df   = pd.DataFrame([flat])
    else:
        raise ValueError(f'Unexpected JSON structure: {type(data)}')

    # Normalise column names
    df = normalise_columns(df)

    # Derive MAP if missing
    if 'map' not in df.columns and \
       'sbp' in df.columns and 'dbp' in df.columns:
        df['map'] = (df['sbp'] + 2 * df['dbp']) / 3

    # Add hours if missing
    if 'hours_from_admission' not in df.columns:
        df['hours_from_admission'] = range(len(df))

    print(f'JSON parsed → {len(df)} rows, columns: {list(df.columns)}')
    return df


def parse_fhir(content: str) -> pd.DataFrame:
    """Parse FHIR R4 Bundle with Observation resources.
    
    Groups observations into hourly rows by:
      1. effectiveDateTime (if present) → extract hour offset
      2. Sequential grouping — every N observations = 1 row
         (where N = number of unique LOINC codes found)
    """
    bundle = json.loads(content)

    # First pass: collect all observations with their codes
    observations = []
    unique_codes = set()

    for entry in bundle.get('entry', []):
        resource = entry.get('resource', {})
        if resource.get('resourceType') != 'Observation':
            continue

        code = None
        for coding in resource.get('code', {}).get('coding', []):
            if 'loinc' in coding.get('system', '').lower():
                code = coding.get('code')
                break

        feature = LOINC_MAP.get(code)
        if not feature:
            continue

        value = (
            resource.get('valueQuantity', {}).get('value') or
            resource.get('valueDecimal') or
            resource.get('valueInteger')
        )
        if value is None:
            continue

        # Try to extract timestamp for grouping
        timestamp = (
            resource.get('effectiveDateTime') or
            resource.get('effectivePeriod', {}).get('start') or
            resource.get('issued')
        )

        unique_codes.add(feature)
        observations.append({
            'feature': feature,
            'value': float(value),
            'timestamp': timestamp,
        })

    if not observations:
        raise ValueError('No valid Observation resources found in FHIR Bundle')

    # ── Strategy 1: Group by timestamp if available ───────────────────────────
    has_timestamps = all(obs['timestamp'] is not None for obs in observations)

    if has_timestamps:
        # Group by timestamp string → each unique timestamp = one row
        from collections import OrderedDict
        time_groups = OrderedDict()
        for obs in observations:
            ts = obs['timestamp']
            if ts not in time_groups:
                time_groups[ts] = {}
            time_groups[ts][obs['feature']] = obs['value']

        rows = []
        for i, (ts, features) in enumerate(time_groups.items()):
            features['hours_from_admission'] = i
            rows.append(features)

        return pd.DataFrame(rows)

    # ── Strategy 2: Sequential grouping by observation count ──────────────────
    # If we have 14 unique observation types and 168 total observations,
    # that's 168/14 = 12 hourly rows
    n_types = len(unique_codes)
    rows = {}
    obs_counter = 0

    for obs in observations:
        hour = obs_counter // n_types
        if hour not in rows:
            rows[hour] = {'hours_from_admission': hour}
        rows[hour][obs['feature']] = obs['value']
        obs_counter += 1

    if not rows:
        raise ValueError('No valid Observation resources found in FHIR Bundle')

    print(f'FHIR parsed → {len(rows)} rows from {len(observations)} observations '
          f'({n_types} unique types)')

    return pd.DataFrame(list(rows.values()))


def parse_hl7(content: str) -> pd.DataFrame:
    """Parse HL7 v2 with OBX segments."""
    rows = {}
    hour = 0

    for line in content.strip().split('\n'):
        segments = line.strip().split('|')
        if not segments:
            continue

        if segments[0] == 'OBR':
            hour += 1

        elif segments[0] == 'OBX' and len(segments) > 5:
            identifier = segments[3].upper().split('^')[0].strip()
            feature    = HL7_MAP.get(identifier)
            if not feature:
                continue
            try:
                value = float(segments[5].split('^')[0])
            except:
                continue

            if hour not in rows:
                rows[hour] = {'hours_from_admission': hour}
            rows[hour][feature] = value

    if not rows:
        raise ValueError('No valid OBX segments found in HL7 file')

    return pd.DataFrame(list(rows.values()))


# ══════════════════════════════════════════════════════════════════════════════
#  FINAL NORMALISATION
# ══════════════════════════════════════════════════════════════════════════════

def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Final normalisation:
    1. Convert all feature cols to numeric
    2. Derive MAP if missing
    3. Sort by time
    4. Forward fill within patient
    5. ★ COMPUTE SOFA SCORES from raw clinical data
    6. Median impute remaining missing
    """
    df = df.copy()

    # Force numeric on all feature cols + extra clinical cols
    all_numeric_cols = FEATURE_COLS + [
        'gcs', 'pao2', 'fio2', 'norepinephrine', 'dopamine',
        'dobutamine', 'epinephrine', 'vasopressor',
        'urine_output', 'urine_output_hourly',
    ]
    for col in all_numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Convert hourly urine output to 24h equivalent for SOFA scoring
    if 'urine_output_hourly' in df.columns and 'urine_output' not in df.columns:
        df['urine_output'] = df['urine_output_hourly'] * 24
        print(f'  ⚕ Converted hourly urine output to 24h '
              f'(e.g. {df["urine_output_hourly"].iloc[0]:.0f} mL/hr '
              f'→ {df["urine_output"].iloc[0]:.0f} mL/day)')
    elif 'urine_output' in df.columns:
        # Auto-detect: if median < 100, likely hourly → convert
        median_uo = df['urine_output'].median()
        if median_uo < 100:
            df['urine_output'] = df['urine_output'] * 24
            print(f'  ⚕ Urine output appears hourly (median={median_uo:.0f}), '
                  f'converted to 24h equivalent')

    # Derive MAP if still missing after normalise_columns
    if 'map' not in df.columns or df['map'].isna().all():
        if 'sbp' in df.columns and 'dbp' in df.columns:
            df['map'] = (df['sbp'] + 2 * df['dbp']) / 3

    # Sort by time
    if 'hours_from_admission' in df.columns:
        df = df.sort_values('hours_from_admission').reset_index(drop=True)

    # Forward fill
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = df[col].ffill()

    # ★ COMPUTE SOFA SCORES from raw clinical data ────────────────────────────
    print('Computing SOFA scores...')
    df = compute_sofa_scores(df)

    # Median impute remaining missing
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = MEDIANS.get(col, 0)
        else:
            df[col] = df[col].fillna(MEDIANS.get(col, 0))

    # Log what we have vs what we expected
    missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_cols:
        print(f'⚠ Imputed missing columns with medians: {missing_cols}')

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  MASTER PARSER
# ══════════════════════════════════════════════════════════════════════════════

def parse_file(filename: str, content: str) -> Tuple[pd.DataFrame, str]:
    """
    Master parser — detects format, parses, normalises.
    Returns (df, format_detected)
    """
    fmt = detect_format(filename, content)
    print(f'Detected format: {fmt} for file: {filename}')

    try:
        if fmt == 'csv':
            df = parse_csv(content)
        elif fmt == 'json':
            df = parse_json(content)
        elif fmt == 'fhir':
            df = parse_fhir(content)
        elif fmt == 'hl7':
            df = parse_hl7(content)
        else:
            # Fallback → try CSV then JSON
            try:
                df  = parse_csv(content)
                fmt = 'csv'
            except:
                df  = parse_json(content)
                fmt = 'json'

    except Exception as e:
        print(f'Parser error ({fmt}): {e}')
        # Last resort fallback
        try:
            df  = parse_csv(content)
            fmt = 'csv-fallback'
        except Exception as e2:
            raise ValueError(
                f'Could not parse file as {fmt}: {e}\n'
                f'CSV fallback also failed: {e2}'
            )

    df = normalise(df)

    print(f'Final DataFrame: {len(df)} rows × {df.shape[1]} cols')
    print(f'Feature coverage: '
          f'{sum(1 for c in FEATURE_COLS if c in df.columns)}'
          f'/{len(FEATURE_COLS)} columns')

    # Log SOFA summary
    sofa_cols = ['sofa_resp', 'sofa_coag', 'sofa_liver',
                 'sofa_cardio', 'sofa_neuro', 'sofa_renal']
    for col in sofa_cols:
        if col in df.columns:
            vals = df[col]
            print(f'  {col}: min={vals.min():.0f} max={vals.max():.0f} '
                  f'mean={vals.mean():.2f} non-zero={int((vals > 0).sum())}')

    return df, fmt