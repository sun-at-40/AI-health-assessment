import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import streamlit as st

# Clinical (low_risk, high_risk) bounds per field, used to normalize mixed-unit
# inputs onto a single 0-100 relative-risk scale for the radar chart.
# invert=True means a LOWER raw value is the riskier end (e.g. low albumin).
RISK_RANGES = {
    'age': (20, 90, False),
    'bmi': (18.5, 40, False),
    'hba1c_level': (4.0, 10.0, False),
    'glucose': (70, 250, False),
    'systolic_bp': (90, 200, False),
    'cholesterol': (150, 300, False),
    'gen_hlth': (1, 5, False),
    'general_health': (1, 5, False),
    'physical_activity': (0, 1, True),
    'phys_activity': (0, 1, True),
    'total_bilirubin': (0.2, 3.0, False),
    'direct_bilirubin': (0.1, 1.5, False),
    'alkaline_phosphotase': (44, 400, False),
    'alamine_aminotransferase': (7, 200, False),
    'aspartate_aminotransferase': (8, 200, False),
    'total_proteins': (4.0, 8.3, True),
    'albumin': (2.0, 5.0, True),
    'albumin_and_globulin_ratio': (0.3, 2.5, True),
    'bp': (60, 180, False),
    'sg': (1.005, 1.025, True),
    'al': (0, 5, False),
    'su': (0, 5, False),
    'bgr': (70, 300, False),
    'bu': (10, 150, False),
    'sc': (0.5, 10, False),
    'sod': (120, 145, True),
    'pot': (3.5, 7.0, False),
    'hemo': (6.0, 17.0, True),
    'pcv': (15, 50, True),
    'wc': (4000, 20000, False),
    'rc': (2.0, 6.0, True),
}

def _normalize_risk(key: str, value: float) -> float:
    """Map a raw clinical value onto a 0-100 relative-risk scale."""
    key_l = key.lower()
    if key_l in RISK_RANGES:
        lo, hi, invert = RISK_RANGES[key_l]
        pct = (value - lo) / (hi - lo) if hi != lo else 0.0
        pct = max(0.0, min(1.0, pct))
        if invert:
            pct = 1 - pct
        return pct * 100
    if value in (0, 1):
        # Binary flags are already coded so that 1 = risk present.
        return value * 100
    # Unknown continuous field: no clinical range available, cap for display only.
    return min(abs(value), 100)

def render_radar_chart(input_data: dict):
    """
    Renders a Radar Chart showing each factor's relative risk (0-100%),
    normalized per-field so mixed units (e.g. glucose vs. BMI vs. binary
    flags) don't distort the shape.
    """
    categories = []
    values = []
    raw_values = []

    for k, v in input_data.items():
        if isinstance(v, (int, float)) and v > 0 and 'gender' not in k.lower():
            categories.append(k.replace('_', ' ').title())
            values.append(_normalize_risk(k, v))
            raw_values.append(v)

    if not categories:
        st.info("Not enough data for Radar Chart")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Current Status',
        line_color='#FF4B4B',
        customdata=raw_values,
        hovertemplate="%{theta}: %{customdata} (%{r:.0f}% risk)<extra></extra>"
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                ticksuffix="%"
            )),
        showlegend=False,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    st.plotly_chart(fig, use_container_width=True)  # Still valid for plotly_chart


def render_trend_chart(records: list, metric_key: str, label: str):
    """
    Renders a Line Chart for a specific metric over time.
    records: List of dicts from api.fetch_records()
    """
    if not records:
        st.info("No historical data for trends.")
        return

    # Extract Data
    dates = []
    values = []
    
    import json
    for r in records:
        try:
            d = json.loads(r['data'])
            if metric_key in d:
                dates.append(r['timestamp'])
                values.append(d[metric_key])
        except:
            continue
            
    if not dates:
        st.warning(f"No data found for {label}")
        return
        
    df = pd.DataFrame({"Date": dates, label: values})
    fig = px.line(df, x="Date", y=label, markers=True, title=f"{label} Over Time")
    fig.update_layout(xaxis_title="Checkup Date", yaxis_title=label)
    
    st.plotly_chart(fig, use_container_width=True)  # Still valid for plotly_chart
