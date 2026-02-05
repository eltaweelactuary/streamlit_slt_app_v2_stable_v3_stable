"""
Konecta SLT: Standalone Desktop Live Analyzer
Real-time sign language recognition using direct camera access.
Run this script locally for the best performance experience.

Usage:
    python standalone_live.py

Controls:
    SPACE - Start/Stop recording
    S     - Speak the recognized sentence (Text-to-Speech)
    Q     - Quit
    R     - Reset sentence
"""

import cv2
import numpy as np
import time
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sign_language_core import SignLanguageCore

# Text-to-Speech setup
try:
    import pyttsx3
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('rate', 150)  # Speed of speech
    TTS_AVAILABLE = True
    print("🔊 Text-to-Speech: Enabled")
except ImportError:
    TTS_AVAILABLE = False
    print("⚠️ Text-to-Speech: Disabled (install pyttsx3: pip install pyttsx3)")

def speak_text(text):
    """Convert text to speech."""
    if TTS_AVAILABLE and text:
        try:
            tts_engine.say(text)
            tts_engine.runAndWait()
            return True
        except Exception as e:
            print(f"❌ TTS Error: {e}")
            return False
    return False

def main():
    print("🚀 Konecta SLT Desktop Live Analyzer")
    print("=" * 50)
    print("Initializing camera and AI engine...")
    
    # Initialize the core engine
    core = SignLanguageCore()
    if not core.is_trained:
        print("⚠️ Training model... (first run only)")
        core.train_core()
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("✅ Camera initialized successfully!")
    print("\n📋 Controls:")
    print("   SPACE - Start/Stop recording")
    print("   Q     - Quit")
    print("   R     - Reset sentence")
    print("\n🎥 Starting live feed...")
    
    # State variables
    recording = False
    recorded_frames = []
    recognized_words = []
    last_prediction = ""
    
    # Create window
    cv2.namedWindow("Konecta SLT Live", cv2.WINDOW_NORMAL)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error reading frame")
            break
        
        # Mirror the frame for natural interaction
        frame = cv2.flip(frame, 1)
        
        # Draw UI overlay
        overlay = frame.copy()
        
        # Status bar at top
        cv2.rectangle(overlay, (0, 0), (640, 60), (40, 40, 40), -1)
        
        # Recording indicator
        if recording:
            cv2.circle(overlay, (30, 30), 12, (0, 0, 255), -1)
            cv2.putText(overlay, f"RECORDING ({len(recorded_frames)} frames)", 
                       (50, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            recorded_frames.append(frame.copy())
        else:
            cv2.circle(overlay, (30, 30), 12, (100, 100, 100), -1)
            cv2.putText(overlay, "Press SPACE to record", 
                       (50, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        # Sentence display at bottom
        cv2.rectangle(overlay, (0, 420), (640, 480), (40, 40, 40), -1)
        sentence = " ".join(recognized_words) if recognized_words else "Recognized signs will appear here..."
        cv2.putText(overlay, sentence[:60], (10, 455), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)
        
        # Last prediction
        if last_prediction:
            cv2.putText(overlay, f"Last: {last_prediction}", (450, 38), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Blend overlay
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Show frame
        cv2.imshow("Konecta SLT Live", frame)
        
        # Handle key input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\n👋 Goodbye!")
            break
        
        elif key == ord(' '):  # SPACE
            if not recording:
                # Start recording
                recording = True
                recorded_frames = []
                print("🔴 Recording started...")
            else:
                # Stop recording and analyze
                recording = False
                print(f"⏹️ Recording stopped. Analyzing {len(recorded_frames)} frames...")
                
                if len(recorded_frames) > 10:
                    # Save to temp file and analyze
                    temp_path = os.path.join(os.path.dirname(__file__), "temp_recording.mp4")
                    h, w = recorded_frames[0].shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(temp_path, fourcc, 20.0, (w, h))
                    for f in recorded_frames:
                        out.write(f)
                    out.release()
                    
                    # Predict
                    labels, conf = core.predict_sentence(temp_path)
                    if labels:
                        last_prediction = " ".join(labels)
                        recognized_words.extend(labels)
                        print(f"✅ Recognized: {last_prediction}")
                    else:
                        print("❌ Could not recognize sign. Try again with clearer gestures.")
                    
                    # Cleanup
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                else:
                    print("⚠️ Recording too short. Hold SPACE longer.")
        
        elif key == ord('r'):
            recognized_words = []
            last_prediction = ""
            print("🔄 Sentence reset")
        
        elif key == ord('s'):
            # Speak the recognized sentence
            if recognized_words:
                sentence_to_speak = " ".join(recognized_words)
                print(f"🔊 Speaking: {sentence_to_speak}")
                speak_text(sentence_to_speak)
            else:
                print("⚠️ No words to speak. Record some signs first.")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
