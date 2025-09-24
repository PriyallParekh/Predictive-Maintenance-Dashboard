import streamlit as st
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
from io import BytesIO
import difflib
import re
import plotly.express as px

# === Set Page Config ===
st.set_page_config(page_title="Upload & Predict Equipment Failure", layout="wide")

# === Load Model ===
@st.cache_resource
def load_model():
    return joblib.load("equipment_rul_predictor.pkl")

model = load_model()

# === Prediction Features ===
features = [
    'Failures_Last_30_Days', 'Failures_Last_60_Days', 'Failures_Last_90_Days',
    'Avg_Downtime_Mins', 'Days_Since_Last_Failure', 'Equipment_Age_Days',
    'Shift_Failure_Count', 'Line_Failure_Count', 'Hour_of_Day', 'Day_of_Week',
    'Month', 'Quarter', 'Rolling_Mean_30d', 'Rolling_Std_30d',
    'Rolling_Mean_90d', 'Rolling_Std_90d', 'Equipment_Health_Score',
    'Maintenance_Count', 'Is_Peak_Hour', 'Is_Weekend',
    'Part_Failure_Frequency', 'Avg_Part_Downtime',
    'Prev_Downtime', 'EMA_Downtime'
]

# === Cleaning Functions ===
def clean_month(month):
    month_map = {
        'jan': '01', 'january': '01', 'januray': '01', 'januay': '01', 'janry': '01', 'jannuary': '01',
        'feb': '02', 'february': '02', 'febuary': '02', 'feburay': '02', 'februry': '02',
        'mar': '03', 'march': '03', 'murch': '03', 'marc': '03', 'mrch': '03',
        'apr': '04', 'april': '04', 'aprl': '04',
        'may': '05',
        'jun': '06', 'june': '06',
        'jul': '07', 'july': '07', 'jly': '07',
        'aug': '08', 'august': '08', 'augst': '08',
        'sep': '09', 'september': '09', 'setember': '09',
        'oct': '10', 'october': '10', 'octobr': '10',
        'nov': '11', 'november': '11', 'novembr': '11',
        'dec': '12', 'december': '12', 'decembr': '12'
    }
    if pd.isna(month):
        return None
    try:
        return month_map[str(month).strip().lower()]
    except:
        return None

def format_date(row):
    try:
        day = int(float(row['date'])) if pd.notna(row['date']) else None
        year = int(float(row['year'])) if pd.notna(row['year']) else None
        month = clean_month(row['month'])
        if day and month and year:
            return f"{day:02d}/{month}/{year}"
        return None
    except:
        return None

def clean_date_columns(df):
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()
    if not all(col in df.columns for col in ['date', 'month', 'year']):
        df['clean_date'] = None
    else:
        df['clean_date'] = df.apply(format_date, axis=1)
    cols = df.columns.tolist()
    if 'clean_date' in cols:
        cols.insert(0, cols.pop(cols.index('clean_date')))
        df = df[cols]
    return df

def map_production_line(machine_id):
    mapping = {
        '1a': 'MLB1-Line1', '1b': 'MLB1-Line2', '2a': 'MLB1-Line3', '2b': 'MLB1-Line4',
        '3a': 'MLB1-Line5', '3b': 'MLB1-Line6', '4a': 'MLB1-Line7', '4b': 'MLB1-Line8',
        '4c': 'MLB1-Line9', 'mlb2 - a': 'MLB2-Line10', 'mlb2 - b': 'MLB2-Line11',
        'mlb2 - c': 'MLB2-Line12', 'mlb2 - d': 'MLB2-Line13', '2a/b': 'MLB1-Line3/MLB1-Line4',
        '2gw': 'MLB1-Line3/MLB1-Line4', '1a & 1b': 'MLB1-Line1/MLB1-Line2', 'both': 'Both', 'all': 'Both'
    }
    if pd.isna(machine_id):
        return None
    try:
        return mapping[str(machine_id).strip().lower()]
    except:
        return None

def clean_production_line(df):
    df = df.copy()
    if 'production_line' not in df.columns:
        df['production_line_standardized'] = None
    else:
        df['production_line_standardized'] = df['production_line'].apply(map_production_line)
    cols = df.columns.tolist()
    if 'production_line' in cols and 'production_line_standardized' in cols:
        prod_line_index = cols.index('production_line')
        cols.insert(prod_line_index + 1, cols.pop(cols.index('production_line_standardized')))
        df = df[cols]
    return df

def clean_equipment_name(df, reference_path="equip.txt"):
    df = df.copy()
    def normalize(name):
        return str(name).strip().lower().replace("-", " ").replace("_", " ").replace("#", " ").replace("  ", " ").strip()
    
    try:
        with open(reference_path, 'r', encoding='utf-8') as f:
            correct_names = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        df['equip_matched'] = 'UNKNOWN'
        return df
    
    normalized_correct_values = list(set(normalize(name) for name in correct_names))
    
    def match_name(raw):
        if pd.isna(raw):
            return "UNKNOWN"
        norm = normalize(raw)
        if norm in normalized_correct_values:
            return norm
        matches = difflib.get_close_matches(norm, normalized_correct_values, n=1, cutoff=0.4)
        return matches[0] if matches else "UNKNOWN"
    
    if 'equipment_name' not in df.columns:
        df['equip_matched'] = 'UNKNOWN'
    else:
        df['equip_matched'] = df['equipment_name'].apply(match_name)
    
    cols = df.columns.tolist()
    if 'equipment_name' in cols and 'equip_matched' in cols:
        equip_index = cols.index('equipment_name')
        cols.insert(equip_index + 1, cols.pop(cols.index('equip_matched')))
        df = df[cols]
    
    return df

def extract_action_and_part(text):
    action_keywords = [
        "replaced", "changed", "serviced", "cleaned", "aligned", "adjusted", "set", "fitted", "mounted", "installed",
        "calibrated", "tightened", "lubricated", "dressed", "reconnected", "connected", "disconnected", "removed",
        "repaired", "fixed", "checked", "inspected", "tested", "trial", "commissioned", "reset", "started", "restarted",
        "origin done", "homed", "parameter set", "updated", "upgraded", "configured", "assembled", "refitted", "fabricated",
        "welded", "soldered", "polished", "greased", "oiled", "rejoined", "positioned", "height set", "air flow set",
        "cable dressed", "dismantled", "tightness done", "servicing done", "function checked", "trial taken", "trial done"
    ]
    part_keywords = [
        "cylinder", "sensor", "belt", "lamp", "magnet", "vacuum cup", "probe", "valve", "bearing", "roller", "filter",
        "seal", "gasket", "o-ring", "cable", "wire", "pipe", "controller", "drive", "motor", "module", "component",
        "assembly", "stud", "spring", "plate", "bolt", "screw", "jaw", "nozzle", "busbar", "frame", "chain", "conveyor",
        "roller", "clamp", "reed switch", "solenoid", "pcb", "smps", "mcb", "encoder", "fan", "cooling fan", "membrane",
        "pad", "pin", "block", "mount", "bracket", "cup", "chamber", "guide", "guide roller", "hook", "stopper",
        "support", "strip", "module", "basket", "cutter", "blade", "press", "pressing pin", "controller", "pcb", "plate",
        "stud", "nut", "washer", "spring", "bearing", "bushing", "grease", "oil", "lubricant", "software", "program",
        "recipe", "parameter", "alignment", "position", "height", "air flow", "cable", "connection", "dressing", "mounting",
        "alignment", "origin", "trial", "calibration", "inspection", "test", "function", "parameter", "setting"
    ]
    if pd.isna(text) or not str(text).strip():
        return "", "UNKNOWN"
    text = str(text).lower().strip()
    steps = re.split(r'[/,;&\n]+', text)
    actions = []
    parts = []
    for step in steps:
        step = step.strip()
        if not step:
            continue
        found_action = next((action.capitalize() for action in action_keywords if action in step), None)
        found_part = next((part for part in part_keywords if part in step), None)
        if found_action:
            actions.append(found_action)
        if found_part:
            parts.append(found_part)
        elif not found_action:
            actions.append(step)
    return ", ".join(actions) if actions else "Uncategorized", ", ".join(parts) if parts else "UNKNOWN"

def clean_action_taken(df):
    df = df.copy()
    if 'action_taken' not in df.columns:
        df['action_taken_standardized'] = ""
        df['modified_part'] = "UNKNOWN"
    else:
        df[['action_taken_standardized', 'modified_part']] = df['action_taken'].apply(
            extract_action_and_part
        ).apply(pd.Series)
    cols = df.columns.tolist()
    if 'action_taken' in cols:
        action_index = cols.index('action_taken')
        cols.insert(action_index + 1, cols.pop(cols.index('action_taken_standardized')))
        cols.insert(action_index + 2, cols.pop(cols.index('modified_part')))
        df = df[cols]
    return df

def clean_types_of_failure(df):
    df = df.copy()
    col = 'types_of_failure(mech/inst/elect/other)'
    if col not in df.columns:
        df['types_of_failure_standardized'] = ""
    else:
        mapping = {
            "oem issue": "oem failure", "oem": "oem failure", "mech": "mechanical failure",
            "mech.": "mechanical failure", "mech ": "mechanical failure", "mechanical": "mechanical failure",
            "machanical": "mechanical failure", "mechanical\"": "mechanical failure",
            "mechanical,_x000d_": "mechanical failure", "\"mechanical,_x000d_": "mechanical failure",
            "inst": "instrumentation failure", "inst.": "instrumentation failure", "instru": "instrumentation failure",
            "instrumentation": "instrumentation failure", "instrumentation\"": "instrumentation failure",
            "\"instrumentation_x000d_": "instrumentation failure", "instrumentation_x000d_": "instrumentation failure",
            "elect": "electrical failure", "elect.": "electrical failure", "elec": "electrical failure",
            "ele": "electrical failure", "electrical": "electrical failure", "other": "other failure",
            "others": "other failure", "oth": "other failure", "other.": "other failure", "open": "other failure",
            "close": "other failure", "closed": "other failure", "opt. issue": "operational failure",
            "opt.issue": "operational failure", "optional issue": "optional failure", "opr. issue": "operational failure",
            "ope. issue": "operational failure", "opration mistake": "operational failure", "opr mistake": "operational failure",
            "software": "software failure", "soft": "software failure", "it": "software failure",
            "spare not available": "spare unavailable", "quality issue": "quality issue", "process": "process failure",
            "plan maintenance": "planned maintenance", "mech/elect": "mechanical /electrical failure",
            "mech + elect": "mechanical /electrical failure", "mech,elec": "mechanical /electrical failure",
            "mech, elect": "mechanical /electrical failure", "mech,elect": "mechanical /electrical failure",
            "elect, mech": "mechanical /electrical failure", "elect , mech": "mechanical /electrical failure",
            "elct,mech": "mechanical /electrical failure", "elec,mech.": "mechanical /electrical failure",
            "elect/mech": "mechanical /electrical failure", "inst/mech": "instrumentation/mechanical failure",
            "inst + mech": "instrumentation/mechanical failure"
        }
        df['types_of_failure_standardized'] = df[col].astype(str).str.strip().str.lower().apply(
            lambda x: mapping.get(x, x) if x else ""
        )
    cols = df.columns.tolist()
    if col in cols:
        failure_index = cols.index(col)
        cols.insert(failure_index + 1, cols.pop(cols.index('types_of_failure_standardized')))
        df = df[cols]
    return df

def categorize_reason(text):
    category_keywords = {
        "Cylinder Failure": ["cylinder", "cyl", "piston", "festo", "leakage", "leackage", "jammed", "not working", "not operating", "shock absorber"],
        "Belt Failure": ["belt", "teflon", "timing belt", "conveyor belt", "welding belt", "sep cell belt", "sap cell belt", "buffing belt", "damage", "damaged", "worn out", "misalignment", "slip", "stuck", "life cycle over"],
        "Magnet Failure": ["magnet", "electromagnet", "magnetic", "stud", "thread damage", "not working", "loose", "bent", "height disturb", "magnetization over"],
        "Sensor Malfunction": ["sensor", "proximity", "photo sensor", "reed switch", "limit switch", "u-type sensor", "optical fiber sensor", "pyrometer", "barcode scanner", "glass presence", "tape detection", "malfunctioning", "not working", "damage", "disturbed", "stuck", "faulty"],
        "Wiring/Cable Failure": ["wire", "cable", "encoder cable", "power cable", "sensor cable", "solenoid coil", "cooling fan wire", "damage", "damaged", "loose", "short", "burn", "came out"],
        "Vacuum System Failure": ["vacuum", "vaccum", "vacuum cup", "vacuum generator", "sucker", "low vacuum", "damage", "damaged", "jammed", "dusty", "height disturb", "position disturb", "leakage"],
        "Alignment Issue": ["alignment", "misalignment", "position disturb", "height disturb", "not ok", "variation", "uneven", "disturbed", "not proper", "incorrectly"],
        "Motor Failure": ["motor", "servo motor", "drive motor", "overheating", "overloading", "position disturb", "error", "jammed", "gear box", "tripped"],
        "Bearing Failure": ["bearing", "linear bearing", "needle roller bearing", "damage", "loose", "noise", "failure"],
        "Valve Failure": ["valve", "solenoid valve", "potting valve", "graco", "leakage", "damaged", "not working", "jammed"],
        "Roller Failure": ["roller", "ribbon roller", "guide roller", "dancing roller", "flatten roller", "cell shaping roller", "worn out", "jammed", "damage", "scratches"],
        "Software Malfunction": ["software", "program", "hmi", "plc", "ccd software", "cws", "corrupt", "corrupted", "hang", "error", "parameter disturb", "recipe not show", "communication lost"],
        "Operator Error": ["operator mistake", "emergency pressed", "incorrectly save", "didn't saw"],
        "Material Jam": ["jam", "jammed", "choked", "stuck", "blockage", "module stuck", "waste plastic"],
        "Frame Issue": ["frame", "frame cross", "frame stuck", "frame shaping", "frame stopper", "frame overlap", "damage", "misalignment", "drop issue"],
        "Lamp Failure": ["lamp", "ir lamp", "xenon lamp", "damage", "broken", "not working", "connection not proper", "life cycle over"],
        "Clamping Issue": ["clamp", "clamping", "tail clamp", "guider clamp", "jaw", "loose", "not proper", "damage", "bent"],
        "Bolt/Nut Failure": ["bolt", "screw", "nut", "mounting bolt", "grub screw", "damage", "loose", "broken", "missing"],
        "Seal/O-Ring Failure": ["seal", "o-ring", "o ring", "membrane", "damage", "damaged", "worn out"],
        "Pulley Failure": ["pulley", "damage", "damaged", "loose", "jammed", "misalignment"],
        "Water Leakage": ["water leakage", "water leak", "pu joint", "cooling line", "water pipe", "water spray", "nozzle height"],
        "Power Supply Issue": ["power trip", "mcb", "smps", "relay", "power socket", "vfd", "ac power", "faulty", "error", "burn"],
        "Blade Failure": ["blade", "trimming blade", "cutter", "cutting", "damage", "loose", "misaligned"],
        "Robot Failure": ["robot", "jt3", "z-axis", "rh pick up", "cover damage", "error", "collision", "not working", "position disturb"],
        "Platform Issue": ["platform", "ribbon support platform", "base plate", "shape plate", "bed", "damage", "misalignment", "height disturb"],
        "Spare Unavailable": ["not available in spare", "spare not available"],
        "Life Cycle Over": ["life cycle over", "lifecycle over"],
        "Temperature Issue": ["temperature out of range", "abnormal temperature"],
        "Chain Failure": ["chain", "drag chain", "damage", "broken"],
        "Laser Failure": ["laser", "groov laser", "groove laser", "not working"],
        "Tape Issue": ["tape", "tap", "taping", "damage", "drop issue", "waste"],
        "Solder Issue": ["solder", "soldering", "dry soldering", "not working", "open soldering"],
        "Other Failure": ["malfunctioning", "unspecified", "unknown", "pm work", "mes problem", "randomly"]
    }
    if pd.isna(text) or not str(text).strip():
        return ""
    text_lower = str(text).lower()
    matches = []
    for category, keywords in category_keywords.items():
        for kw in keywords:
            if re.search(rf'\b{re.escape(kw)}\b', text_lower):
                matches.append(category)
                break
    return ", ".join(sorted(set(matches))) if matches else "Uncategorized"

def clean_reason_for_bd(df):
    df = df.copy()
    if 'resaon_for_bd' not in df.columns:
        df['reason_for_bd_standardized'] = ""
    else:
        df['reason_for_bd_standardized'] = df['resaon_for_bd'].apply(categorize_reason)
    cols = df.columns.tolist()
    if 'resaon_for_bd' in cols:
        reason_index = cols.index('resaon_for_bd')
        cols.insert(reason_index + 1, cols.pop(cols.index('reason_for_bd_standardized')))
        df = df[cols]
    return df

def clean_time_columns(df):
    df = df.copy()
    if not all(col in df.columns for col in ['time_from', 'time_to']):
        df['hour_of_day'] = 0
        df['downtime_mins'] = 0
        return df
    
    def parse_time(time_str, am_pm=None):
        if pd.isna(time_str):
            return None
        try:
            time_str = str(time_str).strip()
            match = re.match(r'(\d{2}:\d{2}:\d{2})', time_str)
            if not match:
                return None
            time_clean = match.group(1)
            if am_pm and not pd.isna(am_pm):
                am_pm = str(am_pm).strip().lower().replace('.', '')
                time_clean += f" {am_pm}"
                return pd.to_datetime(time_clean, format='%H:%M:%S %p', errors='coerce')
            return pd.to_datetime(time_clean, format='%H:%M:%S', errors='coerce')
        except:
            return None
    
    def compute_downtime(row):
        start = row['clean_time_from']
        end = row['clean_time_to']
        if pd.isna(start) or pd.isna(end):
            return 0
        if end < start:
            end += timedelta(days=1)
        return max(0, (end - start).total_seconds() / 60)
    
    if 'am_/_pm' in df.columns:
        df['clean_time_from'] = df.apply(lambda x: parse_time(x['time_from'], x['am_/_pm']), axis=1)
    else:
        df['clean_time_from'] = df['time_from'].apply(parse_time)
    
    df['clean_time_to'] = df['time_to'].apply(parse_time)
    df['hour_of_day'] = df['clean_time_from'].dt.hour.fillna(0).astype(int)
    df['downtime_mins'] = df.apply(compute_downtime, axis=1)
    df = df.drop(['clean_time_from', 'clean_time_to'], axis=1, errors='ignore')
    
    cols = df.columns.tolist()
    if 'time_from' in cols and 'hour_of_day' in cols:
        cols.insert(cols.index('time_from') + 1, cols.pop(cols.index('hour_of_day')))
    if 'time_to' in cols and 'downtime_mins' in cols:
        cols.insert(cols.index('time_to') + 1, cols.pop(cols.index('downtime_mins')))
    df = df[cols]
    
    return df

# === Streamlit App ===
st.title("📁 Upload Maintenance Excel for Failure Prediction")

# === File Upload Section ===
uploaded_file = st.file_uploader("Upload Raw Excel File", type=["xlsx"])
if uploaded_file:
    try:
        raw_df = pd.read_excel(uploaded_file, sheet_name="ATW")
        st.write("✅ File uploaded successfully.")
    except ValueError:
        st.error("❌ Sheet 'breakdown' not found in the uploaded file.")
        st.stop()

    def clean_raw_data(df):
        df.columns = df.columns.str.strip().str.lower().str.replace('\n', ' ').str.replace('"', '').str.replace("  ", " ").str.replace(' ', '_')
        
        df = clean_date_columns(df)
        df = clean_production_line(df)
        df = clean_equipment_name(df)
        df = clean_action_taken(df)
        df = clean_types_of_failure(df)
        df = clean_reason_for_bd(df)
        df = clean_time_columns(df)
        rename_dict = {
            "production_line_standardized": "modified_production_line",
            "equip_matched": "modified_equipment",
        }
        if 'part' in df.columns:
            rename_dict["part"] = "raw_part"
        df = df.rename(columns=rename_dict)
        df["day_of_week"] = pd.to_datetime(df["clean_date"], errors='coerce', dayfirst=True).dt.dayofweek.fillna(0).astype(int)
        df["month"] = pd.to_datetime(df["clean_date"], errors='coerce', dayfirst=True).dt.month.fillna(0).astype(int)
        df["quarter"] = pd.to_datetime(df["clean_date"], errors='coerce', dayfirst=True).dt.quarter.fillna(0).astype(int)
        df["is_weekend"] = df["day_of_week"].isin([5,6]).astype(int)
        df["is_peak_hour"] = df["hour_of_day"].between(8,20).astype(int)
        if 'total_time_(hrs)' in df.columns:
            df['avg_downtime_mins'] = pd.to_numeric(df["total_time_(hrs)"], errors='coerce') * 60
            df['avg_downtime_mins'] = df.apply(
                lambda x: x['downtime_mins'] if pd.isna(x['avg_downtime_mins']) or abs(x['avg_downtime_mins'] - x['downtime_mins']) > 60 else x['avg_downtime_mins'],
                axis=1
            )
        else:
            df['avg_downtime_mins'] = df['downtime_mins']
        df['display_part'] = df.apply(
            lambda x: x['raw_part'] if (pd.isna(x['modified_part']) or x['modified_part'] == 'UNKNOWN') and 'raw_part' in x else x['modified_part'],
            axis=1
        ).fillna('UNKNOWN')
        for col in features:
            if col not in df.columns:
                df[col] = 0
        return df

    try:
        cleaned_df = clean_raw_data(raw_df)
        X_new = cleaned_df[features]
        y_pred = model.predict(X_new)
        cleaned_df["predicted_days_until_failure"] = pd.to_numeric(y_pred.round(0), errors='coerce').astype(int)
        cleaned_df["predicted_failure_date"] = pd.NaT
        valid_mask = cleaned_df["clean_date"].notna() & cleaned_df["predicted_days_until_failure"].notna()
        cleaned_df.loc[valid_mask, "predicted_failure_date"] = (
            pd.to_datetime(cleaned_df.loc[valid_mask, "clean_date"], dayfirst=True, errors='coerce') +
            pd.to_timedelta(cleaned_df.loc[valid_mask, "predicted_days_until_failure"], unit="D")
        )
        st.success("🎯 Prediction complete!")
        # === Define Current Week (June 23–29, 2025) ===
        week_start = "2025-06-23"
        week_end = "2025-06-29"
        # === Filter Data for June 23–29, 2025 ===
        table_df = cleaned_df[
            (cleaned_df["predicted_failure_date"].notna()) &
            (cleaned_df["predicted_failure_date"] >= week_start) & 
            (cleaned_df["predicted_failure_date"] <= week_end)
        ]
        # === Tabs for Overview and Filter ===
        tab1, tab2 = st.tabs(["Overview", "Filter"])
        with tab1:
            st.header("📊 Predicted Failures (June 23–29, 2025)")
            if table_df.empty:
                st.error("No equipment predicted to fail between June 23 and June 29, 2025.")
            else:
                # Production Line Chart
                prod_line_counts = table_df["modified_production_line"].value_counts().reset_index()
                prod_line_counts.columns = ["Production Line", "Failure Count"]
                fig_prod = px.bar(
                    prod_line_counts,
                    x="Production Line",
                    y="Failure Count",
                    title="Predicted Failures by Production Line (June 23–29, 2025)",
                    color="Production Line",
                    text="Failure Count"
                )
                fig_prod.update_layout(showlegend=False, xaxis_tickangle=45)
                st.plotly_chart(fig_prod, use_container_width=True)
                # Indicator Dropdown
                st.subheader("Select Failure Period")
                indicator = st.selectbox(
                    "Choose Indicator",
                    options=["🔴 Recent Failures (This Week)", "🟢 Future Failures (After This Week)"],
                    index=0,
                    key="indicator_dropdown"
                )
                # Line Filter Dropdown
                unique_lines = ['All'] + sorted(table_df['modified_production_line'].dropna().unique().tolist())
                selected_line = st.selectbox("Filter by Line", options=unique_lines, index=0, key="line_filter")
                # Filter Data Based on Indicator
                if indicator == "🔴 Recent Failures (This Week)":
                    filtered_df = cleaned_df[
                        (cleaned_df["predicted_failure_date"].notna()) &
                        (cleaned_df["predicted_failure_date"] >= week_start) &
                        (cleaned_df["predicted_failure_date"] <= week_end)
                    ]
                    title = "Recent Failures (June 23–29, 2025)"
                else:
                    filtered_df = cleaned_df[
                        (cleaned_df["predicted_failure_date"].notna()) &
                        (cleaned_df["predicted_failure_date"] > week_end)
                    ]
                    title = "Future Failures (After June 29, 2025)"
                # Apply Line Filter
                if selected_line != 'All':
                    filtered_df = filtered_df[filtered_df['modified_production_line'] == selected_line]
                # Display Filtered Table
                if filtered_df.empty:
                    st.warning(f"No equipment predicted to fail in the selected period: {title} with line {selected_line if selected_line != 'All' else 'all'}.")
                else:
                    st.subheader(title)
                    # Group by production line, equipment, and date, then aggregate parts
                    aggregated_df = filtered_df.groupby(['modified_production_line', 'modified_equipment', 'predicted_failure_date']).agg({
                        'display_part': lambda x: ', '.join(x.dropna().unique())
                    }).reset_index()
                    aggregated_df["predicted_failure_date"] = aggregated_df["predicted_failure_date"].dt.strftime("%Y-%m-%d")
                    filtered_table = aggregated_df[["modified_production_line", "modified_equipment", "display_part", "predicted_failure_date"]].copy()
                    filtered_table.columns = ["Line", "Equipment Name", "Part", "Predicted Failure Date"]
                    markdown_table = "| Line | Equipment Name | Part | Predicted Failure Date |\n|------|----------------|------|------------------------|\n"
                    for _, row in filtered_table.iterrows():
                        markdown_table += f"| {row['Line']} | {row['Equipment Name']} | {row['Part']} | {row['Predicted Failure Date']} |\n"
                    st.markdown(markdown_table)
                # Equipment Chart (Filter <2)
                equip_counts = table_df["modified_equipment"].value_counts().reset_index()
                equip_counts.columns = ["Equipment", "Failure Count"]
                equip_counts = equip_counts[equip_counts["Failure Count"] >= 2]
                if equip_counts.empty:
                    st.warning("No equipment with 2 or more predicted failures between June 23 and June 29, 2025.")
                else:
                    fig_equip = px.bar(
                        equip_counts,
                        x="Equipment",
                        y="Failure Count",
                        title="Predicted Failures by Equipment (June 23–29, 2025, Count ≥ 2)",
                        color="Equipment",
                        text="Failure Count"
                    )
                    fig_equip.update_layout(showlegend=False, xaxis_tickangle=45)
                    st.plotly_chart(fig_equip, use_container_width=True)
        
        with tab2:
            st.header("📊 Filtered Failures (June 23–29, 2025)")
            if table_df.empty:
                st.error("No equipment predicted to fail between June 23 and June 29, 2025.")
            else:
                # Sidebar Dropdown
                category = st.sidebar.selectbox(
                    "Select Category to Display",
                    options=["Line", "Equipment", "Part"],
                    index=0
                )
                col_map = {
                    "Line": "modified_production_line",
                    "Equipment": "modified_equipment",
                    "Part": "display_part"
                }
                col = col_map[category]
                # Group by production line and equipment to aggregate parts for chart data
                aggregated_df = table_df.groupby(['modified_production_line', 'modified_equipment']).agg({
                    col: lambda x: ', '.join(x.dropna().unique()) if category == "Part" else 'first'
                }).reset_index()
                counts = aggregated_df[col].value_counts().reset_index()
                counts.columns = [category, "Failure Count"]
                fig = px.bar(
                    counts,
                    x=category,
                    y="Failure Count",
                    title=f"Predicted Failures by {category} (June 23–29, 2025)",
                    color=category,
                    text="Failure Count"
                )
                fig.update_layout(showlegend=False, xaxis_tickangle=45)
                st.plotly_chart(fig, use_container_width=True)

        # === Download ===
        output = BytesIO()
        cleaned_df.to_excel(output, index=False)
        st.download_button(
            label="📥 Download Prediction Results",
            data=output.getvalue(),
            file_name="predicted_failures.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception as e:
        st.error(f"❌ Error: {e}")