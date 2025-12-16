import cv2
import os
from mtcnn import MTCNN

# Initialize the Face Detector
detector = MTCNN()

def extract_faces(video_path, save_folder, frame_skip=10):
    # 1. Check if video exists
    if not os.path.exists(video_path):
        print(f"❌ Error: Could not find video at {video_path}")
        return

    # 2. Create save folder if needed
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # 3. Open Video
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0
    video_name = os.path.basename(video_path).split('.')[0]

    print(f"🎥 Processing: {video_name}...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break # Video ended

        # Process every 10th frame (to save time)
        if frame_count % frame_skip == 0:
            try:
                # Detect faces
                faces = detector.detect_faces(frame)
                
                for i, face in enumerate(faces):
                    x, y, width, height = face['box']
                    # Fix coordinates (sometimes MTCNN gives negative numbers)
                    x, y = max(0, x), max(0, y)
                    
                    # Crop face
                    face_img = frame[y:y+height, x:x+width]
                    
                    # Resize to 256x256 (Standard for AI)
                    face_img = cv2.resize(face_img, (256, 256))
                    
                    # Save
                    save_path = os.path.join(save_folder, f"{video_name}_frame{frame_count}_face{i}.jpg")
                    cv2.imwrite(save_path, face_img)
                    saved_count += 1
                    
            except Exception as e:
                pass # Skip errors

        frame_count += 1

    cap.release()
    print(f"✅ Success! Saved {saved_count} face images to: {save_folder}")

# --- RUN THE CODE ---
if __name__ == "__main__":
    # This points to the video you just added
    video_file = "dataset/fake/fake_video.mp4" 
    
    # This is where the face images will go
    output_location = "processed_data/fake"
    
    extract_faces(video_file, output_location)