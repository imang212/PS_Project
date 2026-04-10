import streamlit as st
import pandas as pd
import altair as alt
import os
import json
import cv2

st.set_page_config(page_title="Benchmarky Hailo Pipeline", layout="wide")
st.title("Benchmarky pipeline pro detekci objektů Hailo")

DATA_FILE = "benchmark_results.csv"
JSON_FILE = "benchmark_detections.json"
VIDEO_FILE = "temp_benchmark_video.mp4"

if not os.path.exists(DATA_FILE):
    st.error(f"Datový soubor '{DATA_FILE}' nebyl nalezen. Prosím, nejdříve spusťte benchmarkovací skript.")
else:
    df = pd.read_csv(DATA_FILE)
    
    # Vytvoření jednotného štítku Config pro UI
    df["Config"] = df.apply(
        lambda row: f"{'Sledování' if row['Tracking Enabled'] else 'BezSledování'} | {'Snímky' if row['Frame Materialization'] else 'BezSnímků'}", 
        axis=1
    )

    tab1, tab2, tab3 = st.tabs(["Výkon (FPS)", "Analytika detekcí", "Vizualizace snímků"])

    # ==========================================
    # ZÁLOŽKA 1: VÝKON FPS
    # ==========================================
    with tab1:
        st.markdown("### Porovnání FPS (seskupeno podle modelu)")
        model_order = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']
        
        fps_chart = alt.Chart(df).mark_bar().encode(
            x=alt.X('Config:N', title=None, axis=alt.Axis(labels=False, ticks=False)),
            y=alt.Y('FPS:Q', title='Snímky za sekundu (FPS)'),
            color=alt.Color('Config:N', legend=alt.Legend(title="Konfigurace pipeline", orient="bottom")),
            column=alt.Column('Model:N', sort=model_order, header=alt.Header(title="Velikost modelu", labelOrient='bottom')),
            tooltip=['Model', 'Config', 'FPS', 'Avg Detections']
        ).properties(width=150, height=450)
        
        st.altair_chart(fps_chart, theme="streamlit", use_container_width=False)
        st.dataframe(df, use_container_width=True)

    # ==========================================
    # ZÁLOŽKA 2: ANALYTIKA DETEKCÍ (JSON DATA)
    # ==========================================
    with tab2:
        if not os.path.exists(JSON_FILE):
            st.warning("JSON soubor s daty detekcí nebyl nalezen.")
        else:
            with open(JSON_FILE, 'r') as f:
                det_data = json.load(f)

            # Sestavení DataFrame pro analytiku
            records = []
            frame_records = []
            baseline_config = "Track-False_Frames-False"

            for run_id, frames in det_data.items():
                model = run_id.split('_')[0]
                config = run_id.split('_', 1)[1]
                
                for f_data in frames:
                    # Data na úrovni snímku (pro čárové grafy) - používáme pouze základní config pro zamezení duplicitám
                    if config == baseline_config:
                        confs = [d['confidence'] for d in f_data['detections']]
                        avg_conf = sum(confs) / len(confs) if confs else 0.0
                        frame_records.append({
                            'Model': model,
                            'Frame': f_data['frame_count'],
                            'NumDetections': f_data['num_detections'],
                            'AvgConfidence': avg_conf
                        })

                    # Data na úrovni objektů (pro sloupcové grafy)
                    for d in f_data['detections']:
                        records.append({
                            'Model': model,
                            'Config': config,
                            'Frame': f_data['frame_count'],
                            'Label': d['label'],
                            'Confidence': d['confidence']
                        })
            
            det_df = pd.DataFrame(records)
            frames_df = pd.DataFrame(frame_records)

            if det_df.empty:
                st.info("Nebyly detekovány žádné objekty.")
            else:
                baseline_df = det_df[det_df['Config'] == baseline_config]

                st.markdown("### Časová osa snímek po snímku")
                st.markdown("Sledování stability detekcí a skóre spolehlivosti (confidence) v průběhu videa.")
                
                colA, colB = st.columns(2)
                
                with colA:
                    st.subheader("Detekce v čase")
                    line_det = alt.Chart(frames_df).mark_line().encode(
                        x=alt.X('Frame:Q', title="Číslo snímku"),
                        y=alt.Y('NumDetections:Q', title="Počet detekovaných objektů"),
                        color=alt.Color('Model:N', sort=model_order),
                        tooltip=['Model', 'Frame', 'NumDetections']
                    ).properties(height=300)
                    st.altair_chart(line_det, use_container_width=True)
                    
                with colB:
                    st.subheader("Průměrná spolehlivost v čase")
                    line_conf = alt.Chart(frames_df).mark_line().encode(
                        x=alt.X('Frame:Q', title="Číslo snímku"),
                        y=alt.Y('AvgConfidence:Q', title="Průměrné skóre spolehlivosti", scale=alt.Scale(domain=[0, 1])),
                        color=alt.Color('Model:N', sort=model_order),
                        tooltip=['Model', 'Frame', 'AvgConfidence']
                    ).properties(height=300)
                    st.altair_chart(line_conf, use_container_width=True)

                st.divider()

                st.markdown("### Celková přesnost objektů (Základní konfigurace)")
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Průměrné skóre spolehlivosti")
                    conf_chart = alt.Chart(baseline_df).mark_bar().encode(
                        x=alt.X('Model:N', sort=model_order, axis=alt.Axis(labelAngle=0)),
                        y=alt.Y('mean(Confidence):Q', title='Průměrná spolehlivost', scale=alt.Scale(domain=[0, 1])),
                        color=alt.Color('Model:N', legend=None),
                        tooltip=['Model', 'mean(Confidence)', 'count(Confidence)']
                    ).properties(height=300)
                    st.altair_chart(conf_chart, use_container_width=True)

                with col2:
                    st.subheader("Celkový počet detekovaných objektů (podle třídy)")
                    count_chart = alt.Chart(baseline_df).mark_bar().encode(
                        x=alt.X('Model:N', sort=model_order, axis=alt.Axis(labelAngle=0)),
                        y=alt.Y('count():Q', title='Celkový počet detekovaných boxů'),
                        color=alt.Color('Label:N', legend=alt.Legend(title="Třída objektu")),
                        tooltip=['Model', 'Label', 'count()']
                    ).properties(height=300)
                    st.altair_chart(count_chart, use_container_width=True)

    # ==========================================
    # ZÁLOŽKA 3: VIZUALIZACE SNÍMKŮ
    # ==========================================
    with tab3:
        st.markdown("### Vizuální ověření")
        st.markdown("Vyberte snímek a model, abyste viděli, co přesně AI detekovala. (Používá se základní konfigurace: `BezSledování | BezSnímků`)")

        if not os.path.exists(VIDEO_FILE):
            st.error(f"Video soubor '{VIDEO_FILE}' nebyl nalezen. Ujistěte se, že jej benchmarkovací skript nesmazal.")
        elif not os.path.exists(JSON_FILE):
            st.error("JSON data detekcí nebyla nalezena.")
        else:
            with open(JSON_FILE, 'r') as f:
                det_data = json.load(f)
            
            available_models = sorted(list(set([k.split('_')[0] for k in det_data.keys()])))
            
            max_frame_logged = 1
            for frames in det_data.values():
                if frames:
                    max_frame_logged = max(max_frame_logged, frames[-1]['frame_count'])

            # --- VYLEPŠENÍ UI ZDE ---
            st.markdown("#### Ovládací prvky")
            
            # Použití horizontálního přepínače pro okamžité přepínání
            selected_model = st.radio("Přepnout model:", available_models, horizontal=True)
            
            # Posuvník pro výběr snímku přímo pod přepínačem
            selected_frame = st.slider("Posunout snímek videa:", min_value=1, max_value=max_frame_logged, value=1)
            
            st.divider()
            # ---------------------------

            # Získání detekcí pro vybraný snímek a model
            target_run_id = f"{selected_model}_Track-False_Frames-False"
            frame_detections = []
            
            if target_run_id in det_data:
                frame_data = next((f for f in det_data[target_run_id] if f['frame_count'] == selected_frame), None)
                if frame_data:
                    frame_detections = frame_data['detections']

            # Extrakce snímku pomocí OpenCV
            cap = cv2.VideoCapture(VIDEO_FILE)
            cap.set(cv2.CAP_PROP_POS_FRAMES, selected_frame - 1) 
            ret, frame = cap.read()
            cap.release()

            if not ret:
                st.error("Nepodařilo se načíst tento snímek z video souboru.")
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, _ = frame.shape

                # --- NEW REVERSE LETTERBOX MATH ---
                # 640 is the standard Hailo YOLOv8 input size used in LetterboxAdapter
                target_size = 640.0
                scale = min(target_size / w, target_size / h)
                pad_x = (target_size - w * scale) / 2.0
                pad_y = (target_size - h * scale) / 2.0

                for det in frame_detections:
                    label = det['label']
                    conf = det['confidence']
                    bbox = det['bbox'] # Normalized relative to 640x640
                    
                    # 1. Scale relative to 640x640
                    # 2. Subtract the black bar padding
                    # 3. Divide by the scale factor to return to the original video dimensions
                    x1 = int(((bbox[0] * target_size) - pad_x) / scale)
                    y1 = int(((bbox[1] * target_size) - pad_y) / scale)
                    x2 = int(((bbox[2] * target_size) - pad_x) / scale)
                    y2 = int(((bbox[3] * target_size) - pad_y) / scale)
                    
                    # Clamp values so they don't accidentally draw outside the image bounds
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    # ----------------------------------
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    text = f"{label} {conf:.2f}"
                    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (x1, y1 - 20), (x1 + text_w, y1), (0, 255, 0), -1)
                    cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                st.image(frame, caption=f"Snímek {selected_frame} | Model: {selected_model} | Detekce: {len(frame_detections)}", use_container_width=True)