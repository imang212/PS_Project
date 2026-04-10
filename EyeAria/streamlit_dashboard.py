import streamlit as st
import pandas as pd
import psycopg2
import plotly.express as px
import os

# Match the defaults from PostgresSink.py
DB_HOST = os.environ.get("DB_HOST", "127.0.0.1")
DB_PORT = int(os.environ.get("DB_PORT", 5432))
DB_NAME = os.environ.get("DB_NAME", "eyearia")
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASS = os.environ.get("DB_PASS", "postgres")

# Page config
st.set_page_config(page_title="EyeAria Dashboard", layout="wide", page_icon="👁️")
st.title("EyeAria Pipeline Dashboard")

# 1. Database Connection
@st.cache_resource
def init_connection():
    try:
        return psycopg2.connect(
            host=DB_HOST, port=DB_PORT, dbname=DB_NAME, 
            user=DB_USER, password=DB_PASS
        )
    except Exception as e:
        st.error(f"Failed to connect to DB: {e}")
        return None

conn = init_connection()

if conn is None:
    st.stop()

# 2. Fetch Data Methods
@st.cache_data(ttl=5) # Cache data but refresh every 5 seconds
def get_sessions():
    query = "SELECT * FROM sessions ORDER BY start_time DESC"
    return pd.read_sql(query, conn)

@st.cache_data(ttl=2)
def get_detections(session_id):
    query = f"SELECT * FROM detections WHERE session_id = '{session_id}' ORDER BY timestamp ASC"
    return pd.read_sql(query, conn)

# 3. Sidebar - Session Selector
st.sidebar.header("Filter Data")
sessions_df = get_sessions()

if sessions_df.empty:
    st.warning("No sessions found in the database. Start your EyeAria pipeline to record data!")
    st.stop()

# Format the session dropdown
session_options = sessions_df.apply(
    lambda x: f"{x['model_name']} - {x['pi_uuid']} ({x['start_time']})", axis=1
).tolist()

selected_session_idx = st.sidebar.selectbox(
    "Select Session", 
    range(len(session_options)), 
    format_func=lambda x: session_options[x]
)

selected_session_id = sessions_df.iloc[selected_session_idx]['session_id']

# 4. Main Dashboard Area
st.subheader(f"Session Data: {sessions_df.iloc[selected_session_idx]['model_name']}")

# Fetch detections for the selected session
detections_df = get_detections(selected_session_id)

if detections_df.empty:
    st.info("No detections recorded for this session yet.")
else:
    # --- Metrics Row ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Detections", len(detections_df))
    col2.metric("Unique Objects (Track IDs)", detections_df['track_id'].nunique() if 'track_id' in detections_df else "N/A")
    col3.metric("Average Confidence", f"{detections_df['confidence'].mean() * 100:.1f}%")

    st.markdown("---")

    # --- Charts Row ---
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.write("### Detections by Label")
        label_counts = detections_df['label'].value_counts().reset_index()
        label_counts.columns = ['Label', 'Count']
        fig1 = px.bar(label_counts, x='Label', y='Count', color='Label')
        st.plotly_chart(fig1, use_container_width=True)

    with chart_col2:
        st.write("### Detections Over Time")
        # Group by rounded timestamp to make a clean time-series
        detections_df['time_sec'] = pd.to_datetime(detections_df['timestamp'], unit='s')
        detections_df['time_rounded'] = detections_df['time_sec'].dt.floor('S')
        time_series = detections_df.groupby(['time_rounded', 'label']).size().reset_index(name='count')
        fig2 = px.line(time_series, x='time_rounded', y='count', color='label')
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # --- Raw Data Table ---
    st.write("### Recent Detection Logs")
    # Show the 100 most recent records
    st.dataframe(detections_df.tail(100).sort_values(by='timestamp', ascending=False), use_container_width=True)