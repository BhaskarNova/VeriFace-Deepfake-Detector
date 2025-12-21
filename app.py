import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import tempfile
import time
from mtcnn import MTCNN

# --- PAGE SETUP (Wide Layout & Dark Theme) ---
st.set_page_config(
    page_title="VeriFace: Deepfake Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS (For that "Cyber" look) ---
st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stButton>button {
        background-color: #00ADB5;
        color: white;
        border-radius: 10px;
        height: 50px;
        width: 100%;
        font-weight: bold;
    }
    .stProgress > div > div > div > div {
        background-color: #00ADB5;
    }
    h1 {
        text-align: center; 
        color: #00ADB5;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR (The Control Panel) ---
with st.sidebar:
    st.title("🛡️ VeriFace Control")
    st.markdown("---")
    st.write("This AI uses a **MesoNet** Deep Learning model to detect facial manipulation.")
    
    st.info("💡 **Tip:** Upload a clear video where the face is visible.")
    
    st.markdown("---")
    st.caption("Developed by BhaskarNova")
    st.caption("VeriFace v1.0")

# --- MAIN HEADER ---
st.title("🕵️‍♂️ VeriFace AI Detector")
st.markdown("##### *Analyze videos for deepfake manipulation using Deep Learning*")
st.markdown("---")

# --- LOAD MODEL ---
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('veriface_model.h5')

try:
    with st.spinner("🧠 Loading AI Brain..."):
        model = load_my_model()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Initialize Face Detector
detector = MTCNN()

# --- HELPER FUNCTION ---
def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fake_votes = 0
    real_votes = 0
    frame_count = 0
    
    # Create placeholders for live updates
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Scan every 15th frame (for speed)
        if frame_count % 15 == 0:
            status_text.markdown(f"**Scanning Frame:** `{frame_count}/{total_frames}`")
            progress_bar.progress(min(frame_count / total_frames, 1.0))
            
            try:
                faces = detector.detect_faces(frame)
                for face in faces:
                    x, y, w, h = face['box']
                    x, y = max(0, x), max(0, y)
                    face_img = frame[y:y+h, x:x+w]
                    face_img = cv2.resize(face_img, (256, 256))
                    face_img = face_img / 255.0
                    face_img = np.expand_dims(face_img, axis=0)
                    
                    prediction = model.predict(face_img, verbose=0)
                    score = prediction[0][0]
                    
                    if score > 0.5:
                        real_votes += 1
                    else:
                        fake_votes += 1
            except:
                pass

        frame_count += 1
        
    cap.release()
    progress_bar.empty()
    status_text.empty()
    return real_votes, fake_votes

# --- MAIN INTERFACE (2 Columns) ---
col1, col2 = st.columns([2, 1]) # Left is bigger (Video), Right is smaller (Stats)

with col1:
    st.subheader("📺 Video Feed")
    uploaded_file = st.file_uploader("Upload MP4/AVI", type=["mp4", "mov", "avi"])
    
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        st.video(tfile.name)

with col2:
    st.subheader("📊 Analysis")
    st.write("Waiting for video...")
    
    if uploaded_file is not None:
        if st.button("🚀 Start Analysis"):
            real_count, fake_count = analyze_video(tfile.name)
            
            # RESULTS SECTION
            st.markdown("---")
            total = real_count + fake_count
            
            if total == 0:
                st.warning("⚠️ No faces detected!")
            else:
                # Calculate percentages
                real_score = (real_count / total) * 100
                fake_score = (fake_count / total) * 100
                
                # Display Metrics
                st.metric("Real Confidence", f"{real_score:.1f}%")
                st.metric("Fake Confidence", f"{fake_score:.1f}%")
                
                st.markdown("### Final Verdict:")
                if fake_count > real_count:
                    st.error("🚨 FAKE VIDEO DETECTED")
                else:
                    st.success("✅ REAL VIDEO CONFIRMED")