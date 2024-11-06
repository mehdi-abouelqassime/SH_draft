import streamlit as st
#st.cache_data.clear()
import tempfile
import numpy as np
from roboflow import Roboflow
import cv2
from ultralytics import YOLO


VERSION1 = 2
VERSION2 = 1
rf = Roboflow(api_key="rYoEijlmSCx1L1TuA9ae")  


project = rf.workspace().project("ship-numbers-detection-yauno")  
model = project.version(VERSION1).model 
model1 = YOLO("best (1).pt")


project2 = rf.workspace().project("ship2")  
#model2 = project2.version(VERSION2).model  
model2 = YOLO("best.pt")

st.title("OCP - JORF - SHIP DRAFT WATER DETECTION  ")

uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

min_confidence = st.slider('Min Confidence', 0, 100, 50)
max_overlap = st.slider('Max Overlap', 0, 100, 30)
sampling_time = st.number_input('Sampling Time (in seconds)', min_value=0.1, value=1.0, step=0.1)
show_labels = st.checkbox("Show Labels", value=True)
stroke_width = st.selectbox('Stroke Width', [1, 2, 5, 10], index=1)
tolerance = st.number_input('Y Tolerance', min_value=0.0, value=5.0, step=0.1)

def run_inference_on_frame(model, frame, min_confidence, max_overlap):
    
    temp_image = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(temp_image.name, frame)
    predictions = model.predict(temp_image.name, confidence=min_confidence, overlap=max_overlap).json()
    
    return predictions['predictions']

def run_inference_on_frame_ultralytics(model, frame, min_confidence, max_overlap):
    # Save the frame temporarily to run inference
    temp_image = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(temp_image.name, frame)
    
    class_names = model.names
    
    # Run the YOLO model prediction
    results = model.predict(temp_image.name, conf=min_confidence, iou=max_overlap)
    
    # Prepare predictions in Roboflow-like JSON format
    predictions = []
    for box in results[0].boxes:
        # Extract bounding box details and convert to required format
        x_center, y_center, width, height = box.xywh[0]
        class_id = int(box.cls.item()) if box.cls is not None else -1
        class_name = class_names.get(class_id, "unknown") 
        predictions.append({
            "x": x_center.item(),  # Center x-coordinate
            "y": y_center.item(),  # Center y-coordinate
            "width": width.item(),
            "height": height.item(),
            "class": class_name,  # Class ID as integer
            "confidence": box.conf.item()  # Confidence score
        })
    
    # Clean up the temporary file
    temp_image.close()
    
    # Return the structured predictions
    return  predictions
    

def find_predicted_drift_M(predictions, tolerance):
    """Find the predicted drift mark based on the 'M' label - to modify later."""
    
    # Find the prediction with the minimum y overall
    min_y_pred = None
    b = None 
    for pred in predictions:
        if min_y_pred is None or pred['y'] > min_y_pred['y']:
            min_y_pred = pred
            b = pred

    y=None 
    if b:
        print(b['class'])
        m = str(b['class'])
        print(b)
        m = int(m[:-1])-1
        y = b['y']
    else: 
        m = -1000

        #result_text = f"PREDICTED DRIFT MARK : {m} M, {s}"
    return m,y

    

def find_predicted_drift(predictions, tolerance):
    """Find the predicted drift mark based on the 'small meter numbers' """
    
    # Find the prediction with the minimum y overall
    min_y_pred = None
    b = None 
    for pred in predictions:
        if min_y_pred is None or pred['y'] > min_y_pred['y']:
            if pred['class'] != "M": 
                min_y_pred = pred
                b = pred

    y=None
    if (int(b['class'])==2 or int(b['class'])==4 or int(b['class'])==6 or int(b['class'])==8):
        print(b['class'])
        m = int(b['class'])
        y = b['y']
    else: 
        m = None 
        
        #result_text = f"PREDICTED DRIFT MARK : {m} M, {s}"
    return m,y

if uploaded_file:
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(temp_file.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    sampling_frame_interval = int(sampling_time * fps)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stframe = st.empty()  # Streamlit frame to display results

    frame_count = 0
    processed_frames = 0
    Final_value_list = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % sampling_frame_interval == 0:
            predictions1 = run_inference_on_frame_ultralytics(model1, frame, min_confidence / 100, max_overlap / 100)
            predictions2 = run_inference_on_frame_ultralytics(model2, frame, 30 / 100, max_overlap / 100)
            
            #drift_mark_text, min_y_m_label, a, b = find_predicted_drift(predictions, tolerance)

            mbig,ybig = find_predicted_drift_M(predictions2, tolerance)
            msmall, ysmall = find_predicted_drift(predictions1, tolerance)


            
            
            for pred in predictions2:
                x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
                label = pred['class']
                confidence = pred['confidence']
                
                x1 = int(x - w / 2)
                y1 = int(y - h / 2)
                x2 = int(x + w / 2)
                y2 = int(y + h / 2)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=stroke_width)
                if show_labels:
                    cv2.putText(frame, f"{label} ", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
            for pred in predictions1:
                x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
                label = pred['class']
                confidence = pred['confidence']
                
                x1 = int(x - w / 2)
                y1 = int(y - h / 2)
                x2 = int(x + w / 2)
                y2 = int(y + h / 2)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=stroke_width)
                if show_labels:
                    cv2.putText(frame, f"{label} ", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    
            if mbig != -1000:
                if ysmall!=None: 
                    if ybig>ysmall:
                        mbig = mbig+1
                        msmall = 0
                elif ysmall==None:
                    msmall = 0
                    mbig = mbig+1
                drift_mark_text = f" INSTANT PREDICTED DRIFT MARK : {mbig,msmall}M"
                Final_value_list.append(float(str(str(mbig)+"."+str(msmall))))
                
            else:
                drift_mark_text = "No DETECTION YET" 
            

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, caption=f"Processed Frame {processed_frames} \n {drift_mark_text}", use_column_width=True)
            processed_frames += 1

        frame_count += 1
    Drift = round(np.mean(Final_value_list),1) 
    drift_mark_text = f"FINAL PREDICTED DRIFT MARK : {Drift}M"
    stframe.image(frame_rgb, caption=f"Processed Frame {processed_frames} \n {drift_mark_text}", use_column_width=True)    

    cap.release()
    st.success(f"Finished processing {processed_frames} sampled frames.")
