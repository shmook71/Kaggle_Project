import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer


# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Riyadh Air Traffic Intelligence",
    page_icon="✈️",
    layout="wide",
)

# -----------------------------
# DARK SAAS CSS (clean + cards)
# -----------------------------
DARK_CSS = """
<style>
:root{
  --bg:#0b1020;
  --panel:#111a2e;
  --card:#0f172a;
  --card2:#101b33;
  --border:rgba(255,255,255,.08);
  --text:#e6edf7;
  --muted:rgba(230,237,247,.7);
  --accent:#3b82f6;
  --accent2:#22c55e;
  --warn:#f59e0b;
  --danger:#ef4444;
}

[data-testid="stAppViewContainer"]{
  background: radial-gradient(1200px 600px at 20% 10%, rgba(59,130,246,.20), transparent 60%),
              radial-gradient(1000px 500px at 90% 20%, rgba(34,197,94,.14), transparent 60%),
              var(--bg);
  color: var(--text);
}

[data-testid="stSidebar"]{
  background: linear-gradient(180deg, rgba(17,26,46,.98), rgba(15,23,42,.98));
  border-right: 1px solid var(--border);
}

h1,h2,h3,h4{ color: var(--text); }
p, span, label, div{ color: var(--text); }

.card{
  background: linear-gradient(180deg, rgba(16,27,51,.85), rgba(15,23,42,.85));
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 16px 16px;
  box-shadow: 0 10px 30px rgba(0,0,0,.35);
}

.kpi{
  background: linear-gradient(180deg, rgba(16,27,51,.9), rgba(15,23,42,.9));
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 16px 16px;
  box-shadow: 0 10px 30px rgba(0,0,0,.35);
}

.kpi .label{ font-size: 12px; color: var(--muted); letter-spacing:.3px; }
.kpi .value{ font-size: 24px; font-weight: 700; margin-top: 4px; }
.kpi .hint{ font-size: 12px; color: var(--muted); margin-top: 6px; }

.badge{
  display:inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  border: 1px solid var(--border);
  background: rgba(59,130,246,.12);
  color: var(--text);
  font-size: 12px;
}

hr{ border-color: var(--border); }

[data-testid="stMetricValue"] { color: var(--text) !important; }
[data-testid="stMetricLabel"] { color: var(--muted) !important; }

</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

# -----------------------------
# DATA LOADING
# -----------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Parse times
    for col in ["movement.scheduledTime.utc", "movement.scheduledTime.local"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Standardize key text fields
    for c in [
        "airline.name", "aircraft.model", "status", "flight_type",
        "origin_airport_iata", "destination_airport_iata"
    ]:
        if c in df.columns:
            df[c] = df[c].astype(str).replace({"nan": np.nan, "None": np.nan})

    # Derive time parts (local time is best for business view)
    if "movement.scheduledTime.local" in df.columns:
        df["date_local"] = df["movement.scheduledTime.local"].dt.date
        df["hour_local"] = df["movement.scheduledTime.local"].dt.hour
        df["dow_local"] = df["movement.scheduledTime.local"].dt.day_name()
    else:
        df["date_local"] = pd.NaT
        df["hour_local"] = np.nan
        df["dow_local"] = np.nan

    # Clean status
    if "status" in df.columns:
        df["status_clean"] = (
            df["status"]
            .replace({"nan": np.nan, "None": np.nan})
            .fillna("Unknown")
        )
    else:
        df["status_clean"] = "Unknown"

    return df


def kpi_card(label, value, hint=""):
    st.markdown(
        f"""
        <div class="kpi">
          <div class="label">{label}</div>
          <div class="value">{value}</div>
          <div class="hint">{hint}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(title, subtitle=""):
    st.markdown(
        f"""
        <div class="card">
          <div style="display:flex; align-items:center; justify-content:space-between; gap:12px;">
            <div>
              <div style="font-size:18px; font-weight:700;">{title}</div>
              <div style="font-size:12px; color:rgba(230,237,247,.7); margin-top:4px;">{subtitle}</div>
            </div>
            <span class="badge">RUH • Analytics</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# SIDEBAR NAV + FILTERS
# -----------------------------
st.sidebar.markdown("## ✈️ Riyadh Air Traffic")
st.sidebar.caption("Dark SaaS Intelligence Dashboard")

page = st.sidebar.radio(
    "Navigate",
    ["Dashboard", "Airlines", "Aircraft", "Operations", "ML Insights", "About"],
    index=0
)

data_path = st.sidebar.text_input("Data file", "flights_RUH.csv")
df = load_data(data_path)

st.sidebar.markdown("---")
st.sidebar.markdown("### Filters")

airlines = sorted([a for a in df["airline.name"].dropna().unique().tolist() if a != "Unknown"])
statuses = sorted(df["status_clean"].dropna().unique().tolist())
flight_types = sorted(df["flight_type"].dropna().unique().tolist())

sel_airlines = st.sidebar.multiselect("Airlines", airlines, default=[])
sel_status = st.sidebar.multiselect("Status", statuses, default=[])
sel_ftype = st.sidebar.multiselect("Flight Type", flight_types, default=[])

# Date filter if available
if df["date_local"].notna().any():
    min_d = pd.to_datetime(df["date_local"].min())
    max_d = pd.to_datetime(df["date_local"].max())
    start_d, end_d = st.sidebar.date_input("Date range", (min_d.date(), max_d.date()))
else:
    start_d, end_d = None, None

fdf = df.copy()
if sel_airlines:
    fdf = fdf[fdf["airline.name"].isin(sel_airlines)]
if sel_status:
    fdf = fdf[fdf["status_clean"].isin(sel_status)]
if sel_ftype:
    fdf = fdf[fdf["flight_type"].isin(sel_ftype)]
if start_d and end_d and fdf["date_local"].notna().any():
    fdf = fdf[(pd.to_datetime(fdf["date_local"]) >= pd.to_datetime(start_d)) &
              (pd.to_datetime(fdf["date_local"]) <= pd.to_datetime(end_d))]


# -----------------------------
# DASHBOARD PAGE
# -----------------------------
if page == "Dashboard":
    colA, colB = st.columns([2, 1], gap="large")
    with colA:
        section_title("Dashboard Overview", "KPIs + trends + distribution (filtered view)")
    with colB:
        st.markdown(
            '<div class="card"><div style="font-weight:700;">Quick Notes</div>'
            '<div style="margin-top:8px; color:rgba(230,237,247,.7); font-size:12px;">'
            'Use filters on the left to slice airlines, status, flight type, and date range.'
            '</div></div>',
            unsafe_allow_html=True
        )

    total_flights = len(fdf)
    total_airlines = fdf["airline.name"].nunique(dropna=True)
    total_models = fdf["aircraft.model"].nunique(dropna=True)
    unknown_pct = (fdf["status_clean"].eq("Unknown").mean() * 100) if total_flights else 0

    top_airline = (
        fdf["airline.name"].value_counts(dropna=True).index[0]
        if total_flights and fdf["airline.name"].notna().any()
        else "—"
    )
    top_model = (
        fdf["aircraft.model"].value_counts(dropna=True).index[0]
        if total_flights and fdf["aircraft.model"].notna().any()
        else "—"
    )

    k1, k2, k3, k4, k5, k6 = st.columns(6, gap="medium")
    with k1: kpi_card("Total Flights", f"{total_flights:,}", "records in current filter")
    with k2: kpi_card("Airlines", f"{total_airlines:,}", "unique airline.name")
    with k3: kpi_card("Aircraft Models", f"{total_models:,}", "unique aircraft.model")
    with k4: kpi_card("% Unknown Status", f"{unknown_pct:.1f}%", "data quality indicator")
    with k5: kpi_card("Top Airline", top_airline, "by flight count")
    with k6: kpi_card("Top Aircraft", top_model, "most frequent model")

    st.markdown("")

    c1, c2 = st.columns([1.2, 1], gap="large")

    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Top 10 Airlines")
        top10 = fdf["airline.name"].value_counts(dropna=True).head(10).reset_index()
        top10.columns = ["airline.name", "flights"]
        fig = px.bar(top10, x="airline.name", y="flights")
        fig.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e6edf7"),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Status Distribution")
        sdist = fdf["status_clean"].value_counts(dropna=True).reset_index()
        sdist.columns = ["status", "count"]
        fig2 = px.pie(sdist, names="status", values="count", hole=0.55)
        fig2.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e6edf7"),
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    c3, c4 = st.columns([1, 1], gap="large")

    with c3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Hourly Activity (Local)")
        if fdf["hour_local"].notna().any():
            hour = fdf["hour_local"].value_counts().sort_index().reset_index()
            hour.columns = ["hour", "flights"]
            fig3 = px.line(hour, x="hour", y="flights", markers=True)
            fig3.update_layout(
                height=360,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e6edf7"),
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No local time available to build hourly trend.")
        st.markdown('</div>', unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Top Aircraft Models")
        topm = fdf["aircraft.model"].value_counts(dropna=True).head(10).reset_index()
        topm.columns = ["aircraft.model", "flights"]
        fig4 = px.bar(topm, x="aircraft.model", y="flights")
        fig4.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e6edf7"),
        )
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# -----------------------------
# AIRLINES PAGE
# -----------------------------
elif page == "Airlines":
    section_title("Airlines Analysis", "market share • status quality • routes")
    st.markdown("")

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Airline Market Share")
        share = fdf["airline.name"].value_counts(dropna=True).head(15).reset_index()
        share.columns = ["airline.name", "flights"]
        fig = px.bar(share, x="airline.name", y="flights")
        fig.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e6edf7"),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Unknown Status Rate by Airline (Top 15)")
        tmp = fdf.copy()
        tmp["is_unknown"] = (tmp["status_clean"] == "Unknown").astype(int)
        grp = tmp.groupby("airline.name", dropna=True).agg(
            flights=("flight_number", "count"),
            unknown_rate=("is_unknown", "mean")
        ).reset_index()
        grp = grp.sort_values(["flights"], ascending=False).head(15)
        grp["unknown_rate"] = grp["unknown_rate"] * 100
        fig2 = px.bar(grp, x="airline.name", y="unknown_rate")
        fig2.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e6edf7"),
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# -----------------------------
# AIRCRAFT PAGE
# -----------------------------
elif page == "Aircraft":
    section_title("Aircraft Analysis", "fleet mix • registrations • model distribution")
    st.markdown("")

    c1, c2 = st.columns([1, 1], gap="large")

    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Aircraft Model Distribution (Top 20)")
        mod = fdf["aircraft.model"].value_counts(dropna=True).head(20).reset_index()
        mod.columns = ["aircraft.model", "flights"]
        fig = px.bar(mod, x="aircraft.model", y="flights")
        fig.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e6edf7"),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Top Registrations (aircraft.reg)")
        reg = fdf["aircraft.reg"].value_counts(dropna=True).head(20).reset_index()
        reg.columns = ["aircraft.reg", "flights"]
        fig2 = px.bar(reg, x="aircraft.reg", y="flights")
        fig2.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e6edf7"),
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# -----------------------------
# OPERATIONS PAGE
# -----------------------------
elif page == "Operations":
    section_title("Operations", "routes • origins/destinations • schedule patterns")
    st.markdown("")

    c1, c2 = st.columns([1, 1], gap="large")

    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Top Origins (IATA)")
        o = fdf["origin_airport_iata"].value_counts(dropna=True).head(15).reset_index()
        o.columns = ["origin", "flights"]
        fig = px.bar(o, x="origin", y="flights")
        fig.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e6edf7"),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Top Destinations (IATA)")
        d = fdf["destination_airport_iata"].value_counts(dropna=True).head(15).reset_index()
        d.columns = ["destination", "flights"]
        fig2 = px.bar(d, x="destination", y="flights")
        fig2.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e6edf7"),
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Day of Week Activity (Local)")
    if fdf["dow_local"].notna().any():
        dow = fdf["dow_local"].value_counts().reindex(
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        ).fillna(0).reset_index()
        dow.columns = ["day", "flights"]
        fig3 = px.bar(dow, x="day", y="flights")
        fig3.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e6edf7"),
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No local time available to build day-of-week view.")
    st.markdown('</div>', unsafe_allow_html=True)


# -----------------------------
# ML INSIGHTS PAGE  (FIXED: NaN + option to exclude Unknown)
# -----------------------------
# -----------------------------
# Risk Score (Binary Disruption Model)
# -----------------------------

st.markdown("---")
st.subheader("Risk Score (Disruption Probability)")

# نبني هدف ثنائي
risk_df = fdf.copy()
risk_df["disruption"] = risk_df["status_clean"].apply(
    lambda x: 1 if x in ["Canceled", "CanceledUncertain"] else 0
)

risk_features = [
    "airline.name",
    "aircraft.model",
    "origin_airport_iata",
    "destination_airport_iata",
    "hour_local",
]

risk_features = [c for c in risk_features if c in risk_df.columns]

risk_df = risk_df.dropna(subset=risk_features)

if len(risk_df) > 300:

    X_risk = risk_df[risk_features]
    y_risk = risk_df["disruption"]

    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression

    cat_cols = [c for c in X_risk.columns if X_risk[c].dtype == "object"]
    num_cols = [c for c in X_risk.columns if c not in cat_cols]

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    pre = ColumnTransformer([
        ("cat", cat_pipe, cat_cols),
        ("num", num_pipe, num_cols)
    ])

    risk_model = Pipeline([
        ("pre", pre),
        ("model", LogisticRegression(class_weight="balanced", max_iter=300))
    ])

    risk_model.fit(X_risk, y_risk)

    sample = X_risk.head(200).copy()
    proba = risk_model.predict_proba(sample)[:, 1]

    sample["disruption_proba"] = proba

    def risk_label(p):
        if p >= 0.7:
            return "High"
        elif p >= 0.3:
            return "Medium"
        else:
            return "Low"

    sample["risk_label"] = sample["disruption_proba"].apply(risk_label)

    sample = sample.sort_values("disruption_proba", ascending=False)

    st.dataframe(sample.head(30), use_container_width=True)

else:
    st.warning("Not enough data to build risk model.")
    st.markdown("### Risk Distribution")

risk_dist = sample["risk_label"].value_counts().reset_index()
risk_dist.columns = ["risk_level", "count"]

fig_risk = px.bar(
    risk_dist,
    x="risk_level",
    y="count",
    color="risk_level",
)

fig_risk.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e6edf7")
)

st.plotly_chart(fig_risk, use_container_width=True)