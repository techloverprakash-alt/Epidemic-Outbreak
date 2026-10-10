"""
Epidemic Outbreak Early Warning Radar
Streamlit Dashboard - Interactive Visualization
"""

import streamlit as st  # type: ignore
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import plotly.express as px  # type: ignore
import plotly.graph_objects as go  # type: ignore
from plotly.subplots import make_subplots  # type: ignore
import requests  # type: ignore
import os
import time
import random
from datetime import datetime, timedelta

# ─── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Epidemic Outbreak Early Warning Radar",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Main background */
    .stApp { background: linear-gradient(135deg, #0a0f1e 0%, #0d1b2a 50%, #0a1628 100%); }

    /* Alert banner */
    .alert-banner {
        background: linear-gradient(90deg, #ff2d55, #ff6b35);
        color: white;
        padding: 16px 24px;
        border-radius: 12px;
        font-size: 1.3rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 16px;
        animation: pulse 2s infinite;
        box-shadow: 0 0 30px rgba(255, 45, 85, 0.5);
    }
    @keyframes pulse {
        0%, 100% { box-shadow: 0 0 30px rgba(255, 45, 85, 0.5); }
        50% { box-shadow: 0 0 60px rgba(255, 45, 85, 0.9); }
    }

    /* Safe banner */
    .safe-banner {
        background: linear-gradient(90deg, #00c9a7, #0096ff);
        color: white;
        padding: 16px 24px;
        border-radius: 12px;
        font-size: 1.1rem;
        font-weight: 600;
        text-align: center;
        margin-bottom: 16px;
    }

    /* Metric cards */
    .metric-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        backdrop-filter: blur(10px);
    }

    /* Risk badges */
    .risk-critical { color: #ff2d55; font-weight: 700; }
    .risk-high     { color: #ff9500; font-weight: 700; }
    .risk-moderate { color: #ffcc00; font-weight: 700; }
    .risk-low      { color: #34c759; font-weight: 700; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(10, 25, 50, 0.95) !important;
        border-right: 1px solid rgba(255,255,255,0.1);
    }

    /* Headings */
    h1, h2, h3 { color: #ffffff !important; }
    p, li { color: #a0aec0 !important; }

    /* Streamlit dataframe */
    .stDataFrame { border-radius: 12px; overflow: hidden; }
    /* Map glow effect */
    .map-container {
        border: 2px solid rgba(255, 45, 85, 0.3);
        border-radius: 20px;
        box-shadow: 0 0 40px rgba(255, 45, 85, 0.2);
        overflow: hidden;
        animation: map-pulse 5s infinite;
    }
    @keyframes map-pulse {
        0%, 100% { box-shadow: 0 0 40px rgba(255, 45, 85, 0.2); }
        50% { box-shadow: 0 0 80px rgba(255, 45, 85, 0.4); }
    }
</style>
""", unsafe_allow_html=True)

# ─── API Base URL ──────────────────────────────────────────────────────────────
API_URL = "http://127.0.0.1:5000"

# ─── Helper Functions ──────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def fetch_stats():
    try:
        r = requests.get(f"{API_URL}/api/stats", timeout=20)
        return r.json()
    except Exception:
        return None


@st.cache_data(ttl=30)
def fetch_data(limit=100):
    try:
        r = requests.get(f"{API_URL}/api/data?limit={limit}", timeout=20)
        return r.json()
    except Exception:
        return None


@st.cache_data(ttl=30)
def fetch_locations():
    try:
        r = requests.get(f"{API_URL}/api/locations", timeout=20)
        return r.json()
    except Exception:
        return None


@st.cache_data(ttl=300)
def fetch_ai_report(district):
    try:
        r = requests.get(f"{API_URL}/api/report/{district}", timeout=30)
        return r.json()
    except Exception:
        return None


@st.cache_data(ttl=30)
def fetch_timeline(hours=24):
    try:
        r = requests.get(f"{API_URL}/api/timeline?hours={hours}", timeout=20)
        return r.json()
    except Exception:
        return None


def get_alert_color(risk_level):
    """Return hex color based on risk level."""
    risk_level = str(risk_level).lower()
    if "low" in risk_level:
        return "#28a745"
    elif "moderate" in risk_level:
        return "#ffc107"
    elif "high" in risk_level:
        return "#fd7e14"
    elif "critical" in risk_level:
        return "#dc3545"
    return "#6c757d" # Fallback Gray


def generate_fallback_data():
    """Generate local demo data when Flask API is not running."""
    cities = [
        "Chennai", "Mumbai", "Delhi", "Bangalore", "Kolkata",
        "Hyderabad", "Pune", "Ahmedabad", "Jaipur", "Coimbatore",
    ]
    city_coords = {
        "Chennai": [13.08, 80.27], "Mumbai": [19.07, 72.87],
        "Delhi": [28.70, 77.10], "Bangalore": [12.97, 77.59],
        "Kolkata": [22.57, 88.36], "Hyderabad": [17.38, 78.48],
        "Pune": [18.52, 73.85], "Ahmedabad": [23.02, 72.57],
        "Jaipur": [26.91, 75.78], "Coimbatore": [11.01, 76.95],
        "New York": [40.71, -74.00], "London": [51.50, -0.12],
        "Tokyo": [35.67, 139.65], "Paris": [48.85, 2.35],
        "Berlin": [52.52, 13.40], "Sydney": [-33.86, 151.20],
        "Dubai": [25.20, 55.27], "Singapore": [1.35, 103.81],
        "Geneva": [46.20, 6.14], "Toronto": [43.65, -79.38],
    }
    cities = list(city_coords.keys())
    risk_levels = ["LOW", "LOW", "MODERATE", "MODERATE", "HIGH", "HIGH", "CRITICAL"]
    locations = []
    for city in cities:
        risk = random.choice(risk_levels)
        score = {"LOW": random.uniform(1, 4), "MODERATE": random.uniform(4, 7),
                 "HIGH": random.uniform(7, 12), "CRITICAL": random.uniform(12, 20)}[risk]
        coord = city_coords[city]
        locations.append({
            "city": city, "lat": coord[0], "lon": coord[1],
            "avg_symptom_score": float("{:.2f}".format(score)),
            "max_symptom_score": float("{:.2f}".format(score + random.uniform(1, 4))),
            "post_count": random.randint(10, 200),
            "risk_level": risk, "alert": score >= 6,
        })
    return locations


# ─── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("🔒 **User Session**")
    if "username" not in st.session_state:
        user_in = st.text_input("Username", value="HealthAgent_01")
        phone_in = st.text_input("Mobile Number", placeholder="e.g. 9876543210")
        dist_in = st.selectbox("Your District", [
            "Ariyalur","Chengalpattu","Chennai","Coimbatore","Cuddalore","Dharmapuri",
            "Dindigul","Erode","Kallakurichi","Kanchipuram","Kanyakumari","Karur",
            "Krishnagiri","Madurai","Mayiladuthurai","Nagapattinam","Namakkal","Nilgiris",
            "Perambalur","Pudukkottai","Ramanathapuram","Ranipet","Salem","Sivaganga",
            "Tenkasi","Thanjavur","Theni","Thoothukudi","Tiruchirappalli","Tirunelveli",
            "Tirupathur","Tiruppur","Tiruvallur","Tiruvannamalai","Tiruvarur","Vellore",
            "Viluppuram","Virudhunagar"
        ], index=2)
        if st.button("🔑 Login & Monitor"):
            try:
                requests.post(f"{API_URL}/api/login", json={
                    "username": user_in, 
                    "district": dist_in,
                    "phone": phone_in
                }, timeout=5)
            except Exception:
                pass  # Backend optional for login state
            try:
                # Auto-register with SMS Gateway (optional)
                requests.post("http://localhost:3001/register-agent", json={
                    "username": user_in, "phone": phone_in, "district": dist_in
                }, timeout=2)
            except Exception:
                pass  # SMS Gateway is optional

            st.session_state["username"] = user_in
            st.session_state["user_district"] = dist_in
            st.session_state["user_phone"] = phone_in
            st.rerun()
    else:
        st.success(f"Logged in: {st.session_state['username']}")
        st.info(f"Monitoring: {st.session_state['user_district']}")
        if st.button("🚪 Logout"):
            del st.session_state["username"]
            st.rerun()

    st.markdown("---")
    st.markdown("🏙️ **District Insights (AI)**")
    tn_districts = [
        "Ariyalur","Chengalpattu","Chennai","Coimbatore","Cuddalore","Dharmapuri",
        "Dindigul","Erode","Kallakurichi","Kanchipuram","Kanyakumari","Karur",
        "Krishnagiri","Madurai","Mayiladuthurai","Nagapattinam","Namakkal","Nilgiris",
        "Perambalur","Pudukkottai","Ramanathapuram","Ranipet","Salem","Sivaganga",
        "Tenkasi","Thanjavur","Theni","Thoothukudi","Tiruchirappalli","Tirunelveli",
        "Tirupathur","Tiruppur","Tiruvallur","Tiruvannamalai","Tiruvarur","Vellore",
        "Viluppuram","Virudhunagar"
    ]
    selected_district = st.selectbox("Select District", tn_districts, index=2)
    
    if st.button("📊 Generate AI Report", width="stretch"):
        with st.spinner(f"AI is analyzing {selected_district}..."):
            report = fetch_ai_report(selected_district)
            if report and "error" not in report:
                st.session_state["ai_report"] = report
                st.success(f"Report for {selected_district} ready!")
            else:
                st.error("AI report failed. Check if Gemini API key is set.")

    st.markdown("---")
    st.markdown("🛠️ **Maintenance & Alerts**")
    if st.button("🚨 SMS Alert System", help="Manage District Notifications"):
        st.session_state["show_sms_page"] = True
    
    if st.session_state.get("show_sms_page"):
        st.markdown("---")
        st.markdown("### 🚨 SMS Alert Management")
        
        # Status check
        try:
            r = requests.get("http://localhost:3001/health", timeout=2)
            st.success("✅ SMS Gateway Active")
        except:
            st.error("❌ SMS Server Offline — run: `node backend/sms_server.js`")
        
        tab1, tab2, tab3 = st.tabs(["👤 Register Agent", "📋 Agents List", "🚨 Trigger Alert"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                reg_user = st.text_input("Username", key="sms_reg_user")
            with col2:
                reg_phone = st.text_input("Mobile Number", key="sms_reg_phone", max_chars=10)
            reg_dist = st.selectbox("District", ["Chennai","Coimbatore","Madurai","Salem",
                "Tiruchirappalli","Tirunelveli","Vellore","Erode","Tiruppur","Dindigul",
                "Thanjavur","Dharmapuri","Krishnagiri","Chengalpattu","Kancheepuram",
                "Villupuram","Cuddalore","Nagapattinam","Thanjavur","Vellore"], key="sms_reg_dist")
            if st.button("🔑 Register Agent", key="sms_reg_btn"):
                if reg_user and reg_phone and reg_dist:
                    try:
                        r = requests.post("http://localhost:3001/register-agent", json={
                            "username": reg_user, "phone": reg_phone, "district": reg_dist
                        }, timeout=3)
                        if r.json().get("success"):
                            st.success(f"✅ {reg_user} — {reg_dist} registered!")
                        else:
                            st.error(r.json().get("error"))
                    except:
                        st.error("Server connection failed")
                else:
                    st.warning("All fields required!")
        
        with tab2:
            if st.button("🔄 Refresh", key="sms_refresh"):
                pass
            try:
                r = requests.get("http://localhost:3001/agents", timeout=3)
                data = r.json()
                if data["agents"]:
                    import pandas as pd
                    df = pd.DataFrame(data["agents"])
                    df.columns = ["Username", "Phone", "District"]
                    st.dataframe(df, use_container_width=True)
                    st.info(f"Total: {data['total']} agents")
                else:
                    st.info("No agents registered yet")
            except:
                st.error("Server offline")
        
        with tab3:
            alert_dist = st.selectbox("Select District to Alert", ["Chennai","Coimbatore",
                "Madurai","Salem","Tiruchirappalli","Tirunelveli","Vellore","Erode",
                "Tiruppur","Dindigul","Thanjavur","Dharmapuri","Krishnagiri",
                "Chengalpattu","Kancheepuram"], key="sms_alert_dist")
            custom_msg = st.text_input("Custom Message (optional)", key="sms_custom_msg")
            if st.button("🚨 Send Alert SMS", key="sms_send_btn", type="primary"):
                try:
                    r = requests.post("http://localhost:3001/send-district-alert", json={
                        "district": alert_dist,
                        "message": custom_msg or None
                    }, timeout=10)
                    data = r.json()
                    if data["success"]:
                        st.success(f"✅ {data['sent']}/{data['total']} agents-களுக்கு SMS போச்சு!")
                        for res in data["results"]:
                            if res["success"]:
                                st.write(f"✅ {res['agent']} ({res['phone']})")
                            else:
                                st.write(f"❌ {res['agent']}: {res.get('error','Failed')}")
                    else:
                        st.error(data.get("error"))
                except Exception as e:
                    st.error(f"Error: {e}")
        
        if st.button("✖ Close SMS Panel", key="close_sms"):
            st.session_state["show_sms_page"] = False
            st.rerun()
    
    st.markdown("---")
    st.markdown("### CODE CRAFTERS Team")
    st.markdown("*Hackathon 2K26*")
    st.markdown("---")

    st.markdown("## 🦠 EpiRadar")
    st.markdown("**Epidemic Outbreak Early Warning Radar**")
    st.markdown("---")

    st.markdown("### ⚙️ Controls")
    auto_refresh = st.toggle("🔄 Auto Refresh (30s)", value=False)
    timeline_hours = st.slider("Timeline Hours", 6, 72, 24, step=6)

    st.markdown("---")
    st.markdown("### 🌍 Global Simulation")
    if st.button("⚠️ Trigger Global Pandemic Simulation", width="stretch"):
        try:
            requests.post(f"{API_URL}/api/refresh", json={"outbreak_mode": True, "n": 80}, timeout=20)
            st.cache_data.clear()
            st.success("Outbreak simulation triggered!")
        except Exception:
            st.error("Flask API not running. Showing demo data.")

    if st.button("🔃 Refresh Normal Data", width="stretch"):
        try:
            requests.post(f"{API_URL}/api/refresh", json={"n": 60}, timeout=20)
            st.cache_data.clear()
            st.success("Data refreshed!")
        except Exception:
            st.error("Flask API not running.")

    st.markdown("---")


# ─── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='text-align:center; font-size:2.5rem; font-weight:900;
           background: linear-gradient(90deg, #ff2d55, #ff9500, #ffcc00);
           -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
    🦠 Epidemic Outbreak Early Warning Radar
</h1>
<p style='text-align:center; color:#ff2d55; font-size:1.1rem; font-weight:600; text-transform:uppercase; letter-spacing:2px;'>
    ⚠️ Global Bio-Surveillance Active
</p>
<p style='text-align:center; color:#64748b; font-size:0.9rem;'>
    Real-time Worldwide Disease Monitoring | CODE CRAFTERS | Hackathon 2K26
</p>
""", unsafe_allow_html=True)

# ─── Fetch Data ────────────────────────────────────────────────────────────────
stats = fetch_stats()
data_resp = fetch_data(limit=100)
loc_resp = fetch_locations()
timeline_resp = fetch_timeline(hours=timeline_hours)

api_online = stats is not None

# ─── Global Alert Banner ───────────────────────────────────────────────────────
if api_online:
    # Determine risk level (fallback to LOW if not in stats)
    # We heuristically determine it from global alert and scores if backend doesn't provide it
    global_risk = "LOW"
    if stats:
        avg_score = stats.get("avg_symptom_score", 0)
        is_alert = stats.get("global_alert", False)
        
        if is_alert or avg_score > 10:
            global_risk = "CRITICAL"
        elif avg_score > 7:
            global_risk = "HIGH"
        elif avg_score > 4:
            global_risk = "MODERATE"

    alert_color = get_alert_color(global_risk)
    
    if global_risk in ["HIGH", "CRITICAL"]:
        try:
            requests.post("http://localhost:3001/send-district-alert", json={
                "district": st.session_state.get("user_district", ""),
                "message": f"🚨 {global_risk} RISK ALERT!\nDistrict: {st.session_state.get('user_district','')}\nEpiRadar detected outbreak!\nLogin: localhost:8501"
            }, timeout=5)
        except:
            pass
    
    if global_risk != "LOW":
        st.markdown(f"""
            <div style="background-color: {alert_color}; color: white; padding: 16px 24px; 
                        border-radius: 12px; font-size: 1.3rem; font-weight: 700; 
                        text-align: center; margin-bottom: 16px; box-shadow: 0 4px 15px rgba(0,0,0,0.3);
                        animation: pulse 2s infinite;">
                🚨 ALERT: {global_risk.upper()} RISK — Possible Outbreak Detected!
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div style="background-color: {alert_color}; color: white; padding: 16px 24px; 
                        border-radius: 12px; font-size: 1.1rem; font-weight: 600; 
                        text-align: center; margin-bottom: 16px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                ✅ Monitoring Active — Current Risk: LOW
            </div>
        """, unsafe_allow_html=True)

    # Show SMS Success Message
    if stats and stats.get("sms_sent"):
        st.toast("📲 SMS Alert Sent Successfully to Registered Users", icon="✅")
        st.success("📲 SMS Alert Sent Successfully to Registered Users")
else:
    st.warning("⚠️ Flask API is offline. Showing demo mode data.")

# ─── JS Notification Engine ────────────────────────────────────────────────────
if "username" in st.session_state:
    st.components.v1.html(f"""
    <div id="notification-container" style="position: fixed; top: 20px; right: 20px; z-index: 9999;"></div>
    <script>
    const USER = "{st.session_state['username']}";
    const API_BASE = "{API_URL}";

    // Request browser notification permission
    if (Notification.permission !== "granted") {{
        Notification.requestPermission();
    }}

    function showPopup(msg, level) {{
        const container = document.getElementById('notification-container');
        const alert = document.createElement('div');
        const color = level === 'CRITICAL' ? '#dc3545' : '#fd7e14';
        
        alert.style.cssText = `
            background: ${{color}};
            color: white;
            padding: 20px;
            margin-bottom: 15px;
            border-radius: 12px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.4);
            font-family: 'Inter', sans-serif;
            font-weight: 700;
            width: 350px;
            border-left: 8px solid rgba(0,0,0,0.2);
            animation: slideIn 0.5s ease-out;
            cursor: pointer;
        `;
        
        alert.innerHTML = `<div style="font-size: 0.8rem; opacity: 0.8; margin-bottom: 5px;">📍 PUBLIC HEALTH ALERT</div>${{msg}}`;
        alert.onclick = () => alert.remove();
        
        container.appendChild(alert);
        
        // Browser Push
        if (Notification.permission === "granted") {{
            new Notification("🏥 EpiRadar Status Update", {{ body: msg, icon: "https://cdn-icons-png.flaticon.com/512/2859/2859731.png" }});
        }}

        setTimeout(() => alert.remove(), 10000);
    }}

    let seenAlerts = new Set();

    async function checkAlerts() {{
        try {{
            const res = await fetch(`${{API_BASE}}/api/notifications/${{USER}}`);
            const data = await res.json();
            
            if (data.alerts && data.alerts.length > 0) {{
                data.alerts.forEach(alert => {{
                    if (!seenAlerts.has(alert.id)) {{
                        showPopup(alert.message, alert.level);
                        seenAlerts.add(alert.id);
                    }}
                }});
            }}
        }} catch (e) {{ console.error("Alert check failed", e); }}
    }}

    // Poll every 10 seconds
    setInterval(checkAlerts, 10000);
    checkAlerts(); // Initial check
    
    document.write(`<style>
        @keyframes slideIn {{
            from {{ transform: translateX(120%); opacity: 0; }}
            to {{ transform: translateX(0); opacity: 1; }}
        }}
    </style>`);
    </script>
    """, height=0)

# ─── Row 0: AI District Deep-Dive (Dynamic) ────────────────────────────────────
if "ai_report" in st.session_state:
    st.markdown("---")
    report = st.session_state["ai_report"]
    st.markdown(f"### 🤖 AI Insight: {report['district']} Report")
    
    # 4 Metric columns for AI data
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Primary Disease", report['disease'])
    with m_col2:
        st.metric("Total Cases", report['cases'])
    with m_col3:
        st.metric("Risk Level", report['risk'])
    with m_col4:
        st.metric("Symptom Score", f"{report['symptom_score']}/10")

    # Chart columns
    c_col1, c_col2 = st.columns([2, 1])
    with c_col1:
        st.write("**7-Day Cases Trend**")
        df_ai_tl = pd.DataFrame({"Day": range(1, 8), "Cases": report['timeline']})
        fig_ai_tl = px.line(df_ai_tl, x="Day", y="Cases", markers=True, color_discrete_sequence=["#ff2d55"])
        fig_ai_tl.update_layout(height=250, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_ai_tl, use_container_width=True)
        
    with c_col2:
        st.write("**Disease Breakdown**")
        bd = report['breakdown']
        fig_ai_pie = px.pie(values=list(bd.values()), names=list(bd.keys()), 
                           hole=0.4, color_discrete_sequence=["#ff2d55", "#ff9500", "#0096ff", "#34c759"])
        fig_ai_pie.update_layout(height=250, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
        st.plotly_chart(fig_ai_pie, use_container_width=True)
    
    st.markdown("---")

# ─── KPI Metrics ───────────────────────────────────────────────────────────────
st.markdown("### 📊 Live Dashboard Metrics")
col1, col2, col3, col4, col5 = st.columns(5)

if stats:
    with col1:
        st.metric("📝 Total Posts", stats.get("total_posts", 0))
    with col2:
        st.metric("🚨 Outbreak Posts", stats.get("outbreak_posts", 0),
                  delta=f"{stats.get('outbreak_posts', 0)} flagged", delta_color="inverse")
    with col3:
        st.metric("📈 Avg Symptom Score", stats.get("avg_symptom_score", 0))
    with col4:
        st.metric("🔥 Max Symptom Score", stats.get("max_symptom_score", 0))
    with col5:
        dh = stats.get("disease_hits", {})
        top_disease = max(dh, key=dh.get) if dh else "N/A"
        st.metric("🦠 Top Disease", top_disease)
else:
    with col1: st.metric("📝 Total Posts", "87")
    with col2: st.metric("🚨 Outbreak Posts", "23")
    with col3: st.metric("📈 Avg Symptom Score", "5.4")
    with col4: st.metric("🔥 Max Symptom Score", "16")
    with col5: st.metric("🦠 Top Disease", "Dengue")

st.markdown("---")

# ─── Row 1: Timeline + Disease Distribution ────────────────────────────────────
col_left, col_right = st.columns([2, 1])

with col_left:
    st.markdown("### 📈 Symptom Score Timeline")
    if timeline_resp and "timeline" in timeline_resp:
        tl = timeline_resp["timeline"]
        df_tl = pd.DataFrame(tl)
        fig_tl = go.Figure()
        fig_tl.add_trace(go.Scatter(
            x=df_tl["hour"], y=df_tl["avg_symptom_score"],
            fill="tozeroy", mode="lines+markers",
            line={"color": "#ff2d55", "width": 2.5},
            fillcolor="rgba(255, 45, 85, 0.15)",
            marker={"size": 6, "color": "#ff2d55"},
            name="Avg Symptom Score",
        ))
        # Alert threshold line
        fig_tl.add_hline(y=6, line_dash="dash", line_color="#ffcc00",
                         annotation_text="⚠️ Alert Threshold (6)", annotation_position="top left")
        fig_tl.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#a0aec0"}, margin={"l": 10, "r": 10, "t": 30, "b": 10},
            xaxis={"showgrid": False, "color": "#4a5568"},
            yaxis={"showgrid": True, "gridcolor": "rgba(255,255,255,0.05)", "color": "#4a5568"},
            height=320,
        )
        st.plotly_chart(fig_tl, width="stretch")
    else:
        # Fallback demo timeline
        hours_list = [(datetime.now() - timedelta(hours=i)).strftime("%H:00") for i in range(24, -1, -1)]
        scores = [max(0.5, 3 + 4 * np.sin(i / 4) + random.uniform(-1, 1)) for i in range(25)]
        fig_tl = go.Figure()
        fig_tl.add_trace(go.Scatter(
            x=hours_list, y=scores, fill="tozeroy",
            line={"color": "#ff2d55", "width": 2.5},
            fillcolor="rgba(255, 45, 85, 0.15)",
        ))
        fig_tl.add_hline(y=6, line_dash="dash", line_color="#ffcc00", annotation_text="⚠️ Threshold")
        fig_tl.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#a0aec0"}, margin={"l": 10, "r": 10, "t": 30, "b": 10},
            xaxis={"showgrid": False}, yaxis={"showgrid": True, "gridcolor": "rgba(255,255,255,0.05)"},
            height=320,
        )
        st.plotly_chart(fig_tl, width="stretch")

with col_right:
    st.markdown("### 🦠 Disease Breakdown")
    if stats and stats.get("disease_hits"):
        dh = stats["disease_hits"]
        fig_pie = go.Figure(go.Pie(
            labels=list(dh.keys()),
            values=list(dh.values()),
            hole=0.55,
            marker={"colors": ["#ff2d55", "#ff9500", "#0096ff", "#34c759"]},
        ))
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#a0aec0"}, height=320,
            margin={"l": 10, "r": 10, "t": 30, "b": 10},
            showlegend=True, legend={"font": {"color": "#a0aec0"}},
        )
        st.plotly_chart(fig_pie, width="stretch")
    else:
        fig_pie = go.Figure(go.Pie(
            labels=["COVID-19", "Dengue", "Influenza", "Malaria"],
            values=[28, 34, 22, 16],
            hole=0.55,
            marker={"colors": ["#ff2d55", "#ff9500", "#0096ff", "#34c759"]},
        ))
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", font={"color": "#a0aec0"},
            height=320, margin={"l": 10, "r": 10, "t": 30, "b": 10},
        )
        st.plotly_chart(fig_pie, width="stretch")

# ─── Row 2: Global Risk Map ───────────────────────────────────────────────────
st.markdown("<div class='map-container'>", unsafe_allow_html=True)
st.markdown("### 🗺️ Global Pandemic Risk Map")

locations = loc_resp["locations"] if loc_resp else generate_fallback_data()

if locations:
    df_map = pd.DataFrame(locations)
    color_map = {"CRITICAL": "#ff2d55", "HIGH": "#ff9500", "MODERATE": "#ffcc00", "LOW": "#34c759"}
    df_map["color"] = df_map["risk_level"].map(color_map)
    df_map["size"] = df_map["max_symptom_score"] * 2.5

    # Use a dramatic density map for the 'glow' effect (using Probability Area)
    fig_map = px.density_map(
        df_map,
        lat="lat", lon="lon",
        z="outbreak_probability" if "outbreak_probability" in df_map else "max_symptom_score",
        radius=30,
        center={"lat": 11, "lon": 78.5},
        zoom=6,
        map_style="carto-darkmatter",
        color_continuous_scale=[[0, "rgba(0,0,0,0)"], [0.2, "#ff9500"], [1, "#ff2d55"]],
    )
    
    # Add the scatter points on top for discrete risk levels
    fig_scatter = px.scatter_map(
        df_map,
        lat="lat", lon="lon",
        color="risk_level",
        size="size",
        hover_name="city",
        hover_data={
            "avg_symptom_score": True,
            "max_symptom_score": True,
            "lat": False, "lon": False, "size": False,
        },
        color_discrete_map=color_map,
    )
    
    # Merge the scatter points into the density map
    for trace in fig_scatter.data:
        fig_map.add_trace(trace)

    fig_map.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#a0aec0"}, height=600,
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
        legend={"title": "Risk Level", "font": {"color": "#a0aec0"}, "bgcolor": "rgba(0,0,0,0.5)"},
        coloraxis_showscale=False, # Hide the density scale for cleaner look
    )
    st.plotly_chart(fig_map, width="stretch")
st.markdown("</div>", unsafe_allow_html=True)

# ─── Row 3: Location Bar Chart ────────────────────────────────────────────────
st.markdown("### 📍 City-wise Symptom Scores")
if locations:
    df_bar = pd.DataFrame(locations).sort_values("avg_symptom_score", ascending=True)
    fig_bar = px.bar(
        df_bar, x="avg_symptom_score", y="city", orientation="h",
        color="risk_level",
        color_discrete_map={"CRITICAL": "#ff2d55", "HIGH": "#ff9500",
                            "MODERATE": "#ffcc00", "LOW": "#34c759"},
        text="avg_symptom_score",
    )
    fig_bar.add_vline(x=6, line_dash="dash", line_color="#ff2d55",
                      annotation_text="Alert Threshold")
    fig_bar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#a0aec0"}, height=380,
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
        xaxis={"showgrid": True, "gridcolor": "rgba(255,255,255,0.05)", "color": "#4a5568"},
        yaxis={"showgrid": False, "color": "#4a5568"},
    )
    st.plotly_chart(fig_bar, width="stretch")

# ─── Row 5: Random Forest: Feature Importance Area Show ───────────────────────
st.markdown("### 🌲 Random Forest: Model Effectiveness (Area Show)")
st.markdown("<p style='color:#a0aec0; font-size:0.9rem;'>Global Feature Importance & Decision Influence Area</p>", unsafe_allow_html=True)

# Feature importance data (derived from our RF training)
fi_data = {
    "Feature": ["symptom_score", "post_count", "locations", "day_of_week", "hour", "epidemic", "corona", "outbreak", "pandemic", "malaria"],
    "Importance": [0.3026, 0.1051, 0.0895, 0.0801, 0.0583, 0.0426, 0.0382, 0.0312, 0.0300, 0.0264]
}
df_fi = pd.DataFrame(fi_data).sort_values("Importance", ascending=True)

# Show as a beautiful Area Chart to satisfy 'area show'
fig_area = px.area(
    df_fi, x="Importance", y="Feature", orientation="h",
    color_discrete_sequence=["#ff2d55"]
)
fig_area.update_traces(fillcolor="rgba(255, 45, 85, 0.2)", line_color="#ff2d55")
fig_area.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font={"color": "#a0aec0"}, height=400,
    margin={"l": 0, "r": 20, "t": 20, "b": 0},
    xaxis={"showgrid": True, "gridcolor": "rgba(255,255,255,0.05)", "color": "#4a5568"},
    yaxis={"showgrid": False, "color": "#4a5568"},
)
st.plotly_chart(fig_area, width="stretch")

# ─── Row 6: Recent Posts Table ────────────────────────────────────────────────
st.markdown("### 📋 Recent Flagged Posts")

if data_resp and "data" in data_resp:
    df_posts = pd.DataFrame(data_resp["data"])
    if not df_posts.empty:
        df_display = df_posts[["timestamp", "source", "location", "symptom_score", "risk_level", "text"]].copy()
        df_display["timestamp"] = pd.to_datetime(df_display["timestamp"]).dt.strftime("%H:%M:%S")
        df_display.columns = ["Time", "Source", "Location", "Score", "Risk", "Post"]
        st.dataframe(
            df_display.sort_values("Score", ascending=False).head(20),
            width="stretch", height=300
        )
else:
    st.info("No post data available. Start the Flask API to see live data.")

# ─── Auto Refresh ─────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(30)
    st.cache_data.clear()
    st.rerun()

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#4a5568; font-size:0.85rem;'>"
    "🦠 Epidemic Outbreak Early Warning Radar &nbsp;|&nbsp; "
    "CODE CRAFTERS &nbsp;|&nbsp; Hackathon 2K26 &nbsp;|&nbsp; "
    f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    "</p>",
    unsafe_allow_html=True
)
