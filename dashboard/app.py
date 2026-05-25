# ==============================================================================
# MediVigil
# Coder / Lead Developer: Drupad H
# ==============================================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import glob
import json
from utils_dashboard import parse_logs, get_system_stats

st.set_page_config(page_title="Nurse Command Post", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .reportview-container { background: #0E1117; }
    .stMetric .metric-value { color: #00FF00 !important; }
    .critical-alert { color: #FF0000; font-weight: bold; animation: blinker 1s linear infinite; }
    @keyframes blinker { 50% { opacity: 0; } }
</style>
""", unsafe_allow_html=True)

st.title("🏥 MediVigil Nurse Command Post")
st.markdown("### Real-Time Hospital Patient Facial Monitoring System")

with st.sidebar:
    st.header("Control Panel")
    log_dir = st.text_input("Log Directory", "../logs")
    refresh_rate = st.slider("Refresh Rate (sec)", 1, 10, 2)
    
    st.markdown("---")
    st.subheader("System Status")
    cpu, gpu = get_system_stats()
    st.metric("CPU Usage", f"{cpu}%")
    st.metric("GPU Usage", gpu)
    
df_logs, recent_json = parse_logs(log_dir)

if df_logs is not None and not df_logs.empty:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Events Analyzed", len(df_logs))
    with col2:
        criticals = len(df_logs[df_logs['status'].isin(['CRITICAL_DISTRESS', 'BREATHING_DIFFICULTY', 'NEEDS_HELP'])])
        if criticals > 0:
            st.markdown(f"### <div class='critical-alert'>Urgent Alerts: {criticals}</div>", unsafe_allow_html=True)
        else:
            st.metric("Urgent Alerts", 0)
    with col3:
        drowsy = len(df_logs[df_logs['status'] == 'DROWSY'])
        st.metric("Drowsy/Unconscious", drowsy)
    with col4:
         max_conf = df_logs['confidence'].max()
         st.metric("Max Alert Confidence", f"{max_conf:.1f}%")

    st.markdown("---")
    
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Alert Timeline")
        fig = px.line(df_logs, x='timestamp', y='confidence', color='patient_id', markers=True,
                      color_discrete_sequence=px.colors.qualitative.Set1)
        fig.add_hline(y=75, line_dash="dash", line_color="red", annotation_text="Alert Threshold")
        fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        
    with c2:
        st.subheader("Status Distribution")
        fig2 = px.pie(df_logs, names='status', title='Patient Alerts by Category')
        fig2.update_layout(template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("Head Movement Orbit (Latest Critical Event)")
    
    if recent_json and 'raw_mediguard_data' in recent_json and recent_json['raw_mediguard_data']:
        try:
            pose_data = eval(recent_json['raw_mediguard_data'])
            if pose_data:
                yaws = [p[0] for p in pose_data]
                pitches = [p[1] for p in pose_data]
                rolls = [p[2] for p in pose_data]
                
                fig3 = go.Figure(data=[go.Scatter3d(
                    x=yaws, y=pitches, z=rolls,
                    mode='lines+markers',
                    marker=dict(size=4, color=yaws, colorscale='Viridis', opacity=0.8),
                    line=dict(color='darkblue', width=2)
                )])
                fig3.update_layout(title="6DoF Head Trajectory", template="plotly_dark")
                st.plotly_chart(fig3, use_container_width=True)
        except Exception as e:
            st.error(f"Could not parse orbit data: {e}")
    else:
        st.info("No 3D orbit data available from recent critical events.")
        
    st.subheader("Event Log Database")
    st.dataframe(df_logs.tail(20), use_container_width=True)
    
else:
    st.warning(f"No log files found in {log_dir}. Run the main engine to generate data.")

st.button("Manual Refresh")
