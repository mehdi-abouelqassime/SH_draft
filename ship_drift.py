import streamlit as st
st.cache_data.clear()
import cv2
import tempfile
import numpy as np
from roboflow import Roboflow



VERSION1 = 2
VERSION2 = 1
rf = Roboflow(api_key="rYoEijlmSCx1L1TuA9ae")  # Replace with your Roboflow API key


project = rf.workspace().project("ship-numbers-detection-yauno")  # Replace with your project endpoint
model = project.version(VERSION1).model  # Replace VERSION with the correct version number


project2 = rf.workspace().project("ship2")  # Replace with your project endpoint
model2 = project2.version(VERSION2).model  # Replace VERSION with the correct version number



# Streamlit Web UI Elements
st.title("OCP - JORF - SHIP DRAFT WATER DETECTION  ")

# File uploader for video input
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

# Settings for inference
min_confidence = st.slider('Min Confidence', 0, 100, 50)
max_overlap = st.slider('Max Overlap', 0, 100, 30)
sampling_time = st.number_input('Sampling Time (in seconds)', min_value=0.1, value=1.0, step=0.1)
show_labels = st.checkbox("Show Labels", value=True)
stroke_width = st.selectbox('Stroke Width', [1, 2, 5, 10], index=1)
tolerance = st.number_input('Y Tolerance', min_value=0.0, value=5.0, step=0.1)

def run_inference_on_frame(model, frame, min_confidence, max_overlap):
    """Runs inference using the Roboflow model on a single video frame."""
    
    # Save frame to temporary image file
    temp_image = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(temp_image.name, frame)

    # Use the Roboflow model to predict objects in the frame
    predictions = model.predict(temp_image.name, confidence=min_confidence, overlap=max_overlap).json()

    return predictions['predictions']

def find_predicted_drift_M(predictions, tolerance):
    """Find the predicted drift mark based on the 'M' label and other bounding boxes."""
    
    # Find the prediction with the minimum y overall
    min_y_pred = None
    for pred in predictions:
        if min_y_pred is None or pred['y'] > min_y_pred['y']:
            min_y_pred = pred
            b = pred


    if b:
        m = str(b['class'])
        m = int(m[:-1])-1

        #result_text = f"PREDICTED DRIFT MARK : {m} M, {s}"
        return m
    else:
        return "No matching predictions found.", None, None, None
    

def find_predicted_drift(predictions, tolerance):
    """Find the predicted drift mark based on the 'M' label and other bounding boxes."""
    
    # Find the prediction with the minimum y overall
    min_y_pred = None
    for pred in predictions:
        if min_y_pred is None or pred['y'] > min_y_pred['y']:
            if pred['class'] != "M": 
                min_y_pred = pred
                b = pred


    if b:
        m = int(b['class'])


        #result_text = f"PREDICTED DRIFT MARK : {m} M, {s}"
        return m
    else:
        return "No matching predictions found.", None, None, None

if uploaded_file:
    # Save the uploaded video file to a temporary location
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())
    
    # Load video using OpenCV
    cap = cv2.VideoCapture(temp_file.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    sampling_frame_interval = int(sampling_time * fps)

    # Video parameters
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stframe = st.empty()  # Streamlit frame to display results

    frame_count = 0
    processed_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame based on the sampling time
        if frame_count % sampling_frame_interval == 0:
            # Run inference on the sampled frame
            predictions1 = run_inference_on_frame(model, frame, min_confidence / 100, max_overlap / 100)
            predictions2 = run_inference_on_frame(model2, frame, 20 / 100, max_overlap / 100)
            
            # Find the predicted drift mark
            #drift_mark_text, min_y_m_label, a, b = find_predicted_drift(predictions, tolerance)

            M = find_predicted_drift_M(predictions2, tolerance)
            m = find_predicted_drift(predictions1, tolerance)
            
            # Draw bounding boxes and labels on the frame
            for pred in predictions2:
                x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
                label = pred['class']
                confidence = pred['confidence']
                
                # Convert center x, y, width, height to top-left corner and bottom-right corner for OpenCV rectangle
                x1 = int(x - w / 2)
                y1 = int(y - h / 2)
                x2 = int(x + w / 2)
                y2 = int(y + h / 2)
                
                # Draw the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=stroke_width)
                if show_labels:
                    cv2.putText(frame, f"{label} ", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Convert BGR (OpenCV) to RGB for Streamlit display
            drift_mark_text = f"PREDICTED DRIFT MARK : {M,m}M"

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, caption=f"Processed Frame {processed_frames}\n{drift_mark_text}", use_column_width=True)
            processed_frames += 1

        frame_count += 1

    cap.release()
    st.success(f"Finished processing {processed_frames} sampled frames.")
