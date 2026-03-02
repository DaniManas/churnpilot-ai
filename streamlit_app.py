"""
streamlit_app.py
----------------
Frontend for the ML AI Platform.
Run with:  streamlit run streamlit_app.py

Make sure the FastAPI backend is running first:
  uvicorn app.main:app --reload
"""

import time
import requests
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE  = "http://3.142.131.45:8000"  # e.g. "http://127.0.0.1:8000"
API_KEY   = "dev-key-123"
HEADERS   = {"X-API-Key": API_KEY}
ASSISTANT_PASSCODE = st.secrets.get("ASSISTANT_PASSCODE", "")

st.set_page_config(
    page_title="ML AI Platform",
    page_icon="🤖",
    layout="wide",
)

# ── Sidebar navigation ────────────────────────────────────────────────────────
# Avoid hard failure if external image URL is unavailable in hosted environments.
try:
    st.sidebar.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=60)
except Exception:
    st.sidebar.markdown("## 🤖")
st.sidebar.title("ML AI Platform")
st.sidebar.markdown("Customer Churn Prediction")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    ["🏥 Health", "🔮 Single Prediction", "📦 Batch Scoring", "📊 Monitor", "💬 Assistant"],
    label_visibility="collapsed",
)

# ── Helper ────────────────────────────────────────────────────────────────────
def risk_color(label: str) -> str:
    return {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(label, "⚪")


def api_ok() -> bool:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


# ── PAGE: Health ──────────────────────────────────────────────────────────────
if page == "🏥 Health":
    st.title("🏥 System Health")

    if st.button("Refresh", type="primary"):
        st.rerun()

    try:
        resp = requests.get(f"{API_BASE}/health", timeout=5)
        data = resp.json()

        status = data.get("status", "unknown")
        color  = "🟢" if status == "ok" else "🔴"
        st.subheader(f"{color} Status: **{status.upper()}**")
        st.divider()

        col1, col2, col3 = st.columns(3)
        col1.metric("App Version",   data.get("app_version", "—"))
        col2.metric("Model Version", data.get("model_version", "—"))
        col3.metric("Uptime",        f"{data.get('uptime_seconds', 0):.0f}s")

        col4, col5, col6 = st.columns(3)
        col4.metric("Database",       "✅ Connected" if data.get("db_connected") else "❌ Down")
        col5.metric("RAG Index",      "✅ Ready"     if data.get("rag_index_ready") else "⚠️ Not built")
        col6.metric("Total Requests", data.get("total_requests", 0))

    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot reach the API. Make sure the backend is running:\n\n```\nuvicorn app.main:app --reload\n```")


# ── PAGE: Single Prediction ───────────────────────────────────────────────────
elif page == "🔮 Single Prediction":
    st.title("🔮 Single Customer Prediction")
    st.caption("Fill in the customer details below and hit Predict.")

    with st.form("predict_form"):
        st.subheader("Customer Details")

        col1, col2, col3 = st.columns(3)

        with col1:
            tenure = st.slider("Tenure (months)", 0, 72, 3)
            monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 95.5, step=0.5)
            total_charges   = st.number_input("Total Charges ($)", 0.0, 10000.0, float(tenure * monthly_charges), step=1.0)

        with col2:
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
            payment_method   = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", "Bank transfer", "Credit card"
            ])

        with col3:
            senior_citizen    = st.checkbox("Senior Citizen")
            partner           = st.checkbox("Has Partner")
            dependents        = st.checkbox("Has Dependents")
            phone_service     = st.checkbox("Has Phone Service", value=True)
            paperless_billing = st.checkbox("Paperless Billing", value=True)

        submitted = st.form_submit_button("🔮 Predict Churn", type="primary", use_container_width=True)

    if submitted:
        payload = {
            "tenure": tenure,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "contract": contract,
            "internet_service": internet_service,
            "payment_method": payment_method,
            "senior_citizen": senior_citizen,
            "partner": partner,
            "dependents": dependents,
            "phone_service": phone_service,
            "paperless_billing": paperless_billing,
        }

        with st.spinner("Running prediction..."):
            try:
                resp = requests.post(f"{API_BASE}/predict", json=payload, headers=HEADERS, timeout=10)
                if resp.status_code == 200:
                    result = resp.json()

                    st.divider()
                    st.subheader("Result")

                    icon  = risk_color(result["risk_label"])
                    color = {"High": "red", "Medium": "orange", "Low": "green"}[result["risk_label"]]

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Churn Probability", f"{result['probability']*100:.1f}%")
                    col2.metric("Prediction", "Will Churn" if result["prediction"] == 1 else "Will Stay")
                    col3.metric("Risk Level", f"{icon} {result['risk_label']}")

                    # Gauge chart
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=result["probability"] * 100,
                        number={"suffix": "%"},
                        title={"text": "Churn Probability"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar":  {"color": color},
                            "steps": [
                                {"range": [0, 40],  "color": "#d4edda"},
                                {"range": [40, 70], "color": "#fff3cd"},
                                {"range": [70, 100],"color": "#f8d7da"},
                            ],
                            "threshold": {
                                "line": {"color": "black", "width": 4},
                                "thickness": 0.75,
                                "value": result["probability"] * 100,
                            },
                        },
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)

                    with st.expander("ℹ️ What does this mean?"):
                        if result["risk_label"] == "High":
                            st.warning("🔴 **High Risk** — This customer has a >70% chance of churning. Recommend immediate outreach with a retention offer.")
                        elif result["risk_label"] == "Medium":
                            st.info("🟡 **Medium Risk** — Moderate churn risk (40-70%). Consider a proactive check-in or loyalty incentive.")
                        else:
                            st.success("🟢 **Low Risk** — This customer is likely to stay. No immediate action required.")

                    st.caption(f"Request ID: `{result['request_id']}` | Model: `{result['model_version']}` | Latency: `{result['latency_ms']}ms`")
                else:
                    st.error(f"API error: {resp.json()}")
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot reach the API. Is the backend running?")


# ── PAGE: Batch Scoring ───────────────────────────────────────────────────────
elif page == "📦 Batch Scoring":
    st.title("📦 Batch Scoring")
    st.caption("Upload a CSV with customer records to score them all at once.")

    with st.expander("📋 Required CSV columns"):
        st.code("tenure, monthly_charges, total_charges, contract, internet_service, payment_method, senior_citizen, partner, dependents, phone_service, paperless_billing")
        st.caption("Download the sample file from `data/sample_batch.csv` in the project folder.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df_preview = pd.read_csv(uploaded)
        st.write(f"**Preview** — {len(df_preview)} rows")
        st.dataframe(df_preview.head(5), use_container_width=True)
        uploaded.seek(0)

        if st.button("🚀 Submit for Scoring", type="primary"):
            with st.spinner("Submitting job..."):
                try:
                    resp = requests.post(
                        f"{API_BASE}/batch",
                        files={"file": (uploaded.name, uploaded, "text/csv")},
                        headers=HEADERS,
                        timeout=15,
                    )
                    if resp.status_code == 200:
                        job = resp.json()
                        job_id = job["job_id"]
                        st.success(f"✅ Job submitted! ID: `{job_id}`")
                        st.session_state["batch_job_id"] = job_id
                    else:
                        st.error(f"Submission failed: {resp.json()}")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot reach the API.")

    # ── Poll job status ────────────────────────────────────────────────────────
    if "batch_job_id" in st.session_state:
        job_id = st.session_state["batch_job_id"]
        st.divider()
        st.subheader(f"Job Status — `{job_id[:16]}...`")

        try:
            status_resp = requests.get(f"{API_BASE}/batch/{job_id}", headers=HEADERS, timeout=5)
            status_data = status_resp.json()
            status      = status_data["status"]

            col1, col2, col3 = st.columns(3)
            col1.metric("Status", status.upper())
            col2.metric("Processed", f"{status_data.get('processed_records', 0)} / {status_data.get('total_records', '?')}")
            col3.metric("Submitted", status_data.get("created_at", "—")[:19].replace("T", " "))

            if status == "processing":
                st.progress(0.5, text="Processing in background...")
                if st.button("🔄 Refresh Status"):
                    st.rerun()

            elif status == "done":
                st.success("✅ Scoring complete!")
                dl_resp = requests.get(f"{API_BASE}/batch/{job_id}/download", headers=HEADERS, timeout=10)
                if dl_resp.status_code == 200:
                    result_df = pd.read_csv(__import__("io").StringIO(dl_resp.text))

                    # Summary chart
                    risk_counts = result_df["risk_label"].value_counts().reset_index()
                    risk_counts.columns = ["Risk Level", "Count"]
                    color_map = {"High": "#dc3545", "Medium": "#ffc107", "Low": "#28a745"}
                    fig = px.bar(
                        risk_counts, x="Risk Level", y="Count",
                        color="Risk Level", color_discrete_map=color_map,
                        title="Risk Distribution",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    col1, col2, col3 = st.columns(3)
                    col1.metric("🔴 High Risk",   int((result_df["risk_label"] == "High").sum()))
                    col2.metric("🟡 Medium Risk", int((result_df["risk_label"] == "Medium").sum()))
                    col3.metric("🟢 Low Risk",    int((result_df["risk_label"] == "Low").sum()))

                    st.download_button(
                        "⬇️ Download Scored CSV",
                        data=dl_resp.content,
                        file_name=f"scored_{job_id[:8]}.csv",
                        mime="text/csv",
                        type="primary",
                    )
                    st.dataframe(result_df.head(10), use_container_width=True)

            elif status == "failed":
                st.error(f"❌ Job failed: {status_data.get('error_message', 'Unknown error')}")

            elif status == "pending":
                st.info("⏳ Job is queued, starting soon...")
                if st.button("🔄 Refresh Status"):
                    st.rerun()

        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot reach the API.")


# ── PAGE: Monitor ─────────────────────────────────────────────────────────────
elif page == "📊 Monitor":
    st.title("📊 Model Monitoring")
    st.caption("Drift detection and prediction distribution over time.")

    window = st.slider("Analysis window (days)", 1, 90, 7)

    if st.button("Load Report", type="primary"):
        with st.spinner("Fetching monitoring data..."):
            try:
                resp = requests.get(
                    f"{API_BASE}/monitor",
                    params={"window_days": window},
                    headers=HEADERS,
                    timeout=10,
                )
                data = resp.json()

                dist   = data["prediction_distribution"]
                drift  = data["feature_drift"]
                status = data["overall_drift_status"]

                # Overall status banner
                status_display = {"stable": ("🟢", "success"), "warning": ("🟡", "warning"), "drift": ("🔴", "error")}
                icon, stype = status_display.get(status, ("⚪", "info"))
                getattr(st, stype)(f"{icon} Overall drift status: **{status.upper()}**")

                st.divider()
                st.subheader("Prediction Distribution")

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Predictions", dist["total_predictions"])
                col2.metric("🔴 High Risk",      dist["high_risk_count"])
                col3.metric("🟡 Medium Risk",    dist["medium_risk_count"])
                col4.metric("🟢 Low Risk",       dist["low_risk_count"])

                col5, col6 = st.columns(2)
                col5.metric("Churn Count",    dist["churn_count"])
                col6.metric("Avg Probability", f"{dist['avg_probability']*100:.1f}%")

                if dist["total_predictions"] > 0:
                    fig = px.pie(
                        values=[dist["high_risk_count"], dist["medium_risk_count"], dist["low_risk_count"]],
                        names=["High", "Medium", "Low"],
                        color_discrete_map={"High": "#dc3545", "Medium": "#ffc107", "Low": "#28a745"},
                        title="Risk Distribution",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    churn_series = data.get("churn_rate_over_time", [])
                    if churn_series:
                        trend_df = pd.DataFrame(churn_series)
                        trend_df["churn_rate_pct"] = trend_df["churn_rate"] * 100
                        fig_trend = px.line(
                            trend_df,
                            x="date",
                            y="churn_rate_pct",
                            markers=True,
                            title="Churn Rate Over Time",
                            labels={"date": "Date", "churn_rate_pct": "Churn Rate (%)"},
                        )
                        fig_trend.update_yaxes(range=[0, 100])
                        st.plotly_chart(fig_trend, use_container_width=True)

                # PSI drift scores
                if drift:
                    st.divider()
                    st.subheader("Feature Drift (PSI Scores)")
                    st.caption("PSI < 0.10 = stable | 0.10–0.20 = warning | > 0.20 = drift")

                    drift_df = pd.DataFrame(drift)
                    color_map_psi = {"stable": "#28a745", "warning": "#ffc107", "drift": "#dc3545"}

                    fig2 = px.bar(
                        drift_df, x="feature", y="psi_score",
                        color="status", color_discrete_map=color_map_psi,
                        title="PSI Score per Feature",
                        text="psi_score",
                    )
                    fig2.add_hline(y=0.10, line_dash="dash", line_color="orange", annotation_text="Warning threshold")
                    fig2.add_hline(y=0.20, line_dash="dash", line_color="red",    annotation_text="Drift threshold")
                    fig2.update_traces(texttemplate="%{text:.3f}", textposition="outside")
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.info("No drift data yet — make some predictions first to populate the monitor.")

            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot reach the API.")


# ── PAGE: Assistant ───────────────────────────────────────────────────────────
elif page == "💬 Assistant":
    st.title("💬 Knowledge Assistant")
    st.caption("Ask anything about the platform — the model, the API, monitoring, or how churn prediction works.")

    # Optional passcode gate for the Assistant page (set via Streamlit secrets)
    if ASSISTANT_PASSCODE:
        if "assistant_unlocked" not in st.session_state:
            st.session_state.assistant_unlocked = False

        if not st.session_state.assistant_unlocked:
            st.warning("🔒 Assistant access is protected. Enter the passcode to ask questions.")
            st.caption("Need access? Contact the project owner for the Assistant passcode.")
            entered = st.text_input("Assistant passcode", type="password")
            if st.button("Unlock Assistant", type="primary"):
                if entered == ASSISTANT_PASSCODE:
                    st.session_state.assistant_unlocked = True
                    st.rerun()
                else:
                    st.error("Incorrect passcode.")
            st.stop()

    # Init chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                with st.expander("📚 Sources"):
                    for c in msg["citations"]:
                        st.caption(f"**{c['source']}** (score: {c['score']:.2f}) — {c['snippet'][:100]}...")

    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        # Show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base..."):
                try:
                    resp = requests.post(
                        f"{API_BASE}/assist",
                        json={"question": prompt, "top_k": 5},
                        headers=HEADERS,
                        timeout=15,
                    )
                    if resp.status_code == 200:
                        result = resp.json()
                        answer = result["answer"]
                        st.markdown(answer)

                        if result.get("citations"):
                            with st.expander("📚 Sources"):
                                for c in result["citations"]:
                                    st.caption(f"**{c['source']}** (score: {c['score']:.2f}) — {c['snippet'][:100]}...")

                        st.caption(f"Retrieval: {result['retrieval_latency_ms']}ms | Model: {result['model_used']}")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "citations": result.get("citations", []),
                        })
                    else:
                        err = f"API error {resp.status_code}: {resp.json()}"
                        st.error(err)
                        st.session_state.messages.append({"role": "assistant", "content": err})
                except requests.exceptions.ConnectionError:
                    msg = "❌ Cannot reach the API. Make sure the backend is running."
                    st.error(msg)
                    st.session_state.messages.append({"role": "assistant", "content": msg})

    if st.session_state.messages:
        if st.button("🗑️ Clear chat"):
            st.session_state.messages = []
            st.rerun()
