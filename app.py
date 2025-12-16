import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import tempfile
import os
from mtcnn import MTCNN

# --- PAGE SETUP ---
st.set_page_config(page_title="VeriFace: Deepfake Detector", page_icon="🕵️‍♂️")

st.title("🕵️‍♂️ VeriFace")
st.write("Upload a video to check if it's **REAL** or **FAKE**.")

# --- LOAD MODEL (Cache it so it doesn't reload every time) ---
@st.cache_resource
def load_my_model():
    print("Loading model...")
    return tf.keras.models.load_model('veriface_model.h5')

try:
    model = load_my_model()
    st.success("✅ AI Model Loaded Successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Initialize Face Detector
detector = MTCNN()

# --- HELPER FUNCTION: PREDICT VIDEO ---
def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    fake_votes = 0
    real_votes = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Analyze every 15th frame to speed it up
        if frame_count % 15 == 0:
            status_text.text(f"Processing frame {frame_count}/{total_frames}...")
            progress_bar.progress(min(frame_count / total_frames, 1.0))
            
            try:
                # 1. Detect Face
                faces = detector.detect_faces(frame)
                
                for face in faces:
                    x, y, w, h = face['box']
                    x, y = max(0, x), max(0, y)
                    
                    # 2. Crop & Resize
                    face_img = frame[y:y+h, x:x+w]
                    face_img = cv2.resize(face_img, (256, 256))
                    
                    # 3. Normalize (0-1) like we did in training
                    face_img = face_img / 255.0
                    face_img = np.expand_dims(face_img, axis=0) # Add batch dimension
                    
                    # 4. Predict
                    prediction = model.predict(face_img, verbose=0)
                    score = prediction[0][0] # 0 to 1
                    
                    # IMPORTANT: Check your training class indices!
                    # Usually: 0=Fake, 1=Real (Alphabetical order)
                    if score > 0.5:
                        real_votes += 1
                    else:
                        fake_votes += 1
                        
            except Exception as e:
                pass # Skip errors on blurry frames

        frame_count += 1
        
    cap.release()
    progress_bar.empty()
    return real_votes, fake_votes

# --- UI LOGIC ---
uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # Save uploaded file to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    st.video(tfile.name) # Show the video
    
    if st.button("🔍 Analyze Video"):
        with st.spinner("AI is analyzing faces..."):
            real_count, fake_count = analyze_video(tfile.name)
            
        st.write("---")
        st.subheader("Analysis Results")
        col1, col2 = st.columns(2)
        col1.metric("Real Frames Detected", real_count)
        col2.metric("Fake Frames Detected", fake_count)
        
        total = real_count + fake_count
        if total == 0:
            st.warning("No faces detected in the video.")
        else:
            # Final Verdict
            if fake_count > real_count:
                st.error("🚨 VERDICT: FAKE VIDEO DETECTED!")
            else:
                st.success("✅ VERDICT: REAL VIDEO")