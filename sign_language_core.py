"""
Konecta AI Signature Series: Sign Language Core Engine v2.0
Highly optimized for real-time skeletal DNA synthesis and bidirectional translation.
"""

import os
import cv2
import numpy as np
import pickle
import mediapipe as mp
import speech_recognition as sr
from pathlib import Path

class SignLanguageCore:
    """
    Enterprise-Grade Engine for Sign Language Analytics and Synthesis.
    
    This core provides a unified abstraction layer for:
    - Skeletal DNA (CLR) extraction and management.
    - Motion-to-Text classification via Augmented Random Forest Matrices.
    - Benchmarking and vocabulary synchronization.
    """
    def __init__(self, data_dir="./slt_core_assets"):
        """
        Initializes the SLT Core with persistence pathing.
        :param data_dir: Local or transient directory for asset storage.
        """
        self.data_dir = Path(data_dir)
        self.videos_dir = self.data_dir / "videos"
        self.landmarks_dir = self.data_dir / "landmarks"
        self.models_dir = self.data_dir / "models" 
        self.model_path = self.models_dir / "clr_model_v2.pkl"
        
        # Create directories
        try:
            self.videos_dir.mkdir(parents=True, exist_ok=True)
            self.landmarks_dir.mkdir(parents=True, exist_ok=True)
            self.models_dir.mkdir(parents=True, exist_ok=True)
        except:
            pass
        
        # MediaPipe Setup
        self.mp_holistic = mp.solutions.holistic
        
        # Dictionary and Model
        self.landmark_dict = {}
        self.classifier = None
        self.label_encoder = None
        
        # Core Vocabulary (Stable 8 - Confirmed SLT Library Mappings)
        # These words have verified labels in pk-dictionary-mapping.json
        self.vocabulary = {
            "apple": "سیب",      # pk-hfad-1_apple
            "world": "دنیا",     # pk-hfad-1_world  
            "good": "اچھا",      # pk-hfad-1_good
            "school": "اسکول",   # pk-hfad-1_school
            "mother": "ماں",     # pk-hfad-1_mother
            "father": "باپ",     # pk-hfad-1_papa
            "help": "مدد",       # pk-hfad-1_help
            "home": "گھر",       # pk-hfad-1_house
        }
        
        # Core Vocabulary (Stable 8 - Confirmed SLT Library Mappings)
        self.vocabulary = {
            "apple": "سیب", "world": "دنیا", "good": "اچھا",
            "school": "اسکول", "mother": "ماں", "father": "باپ",
            "help": "مدد", "home": "گھر", "is": "ہے"
        }

    def extract_landmarks_from_video(self, video_path, max_frames=60, return_sequence=True):
        """
        Extracts high-fidelity skeletal landmark sequences using MediaPipe Holistic.
        Returns a normalized temporal matrix (N_Frames x M_Features).
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"🚫 Could not open video file: {video_path}")
            return None
            
        features_sequence = []
        
        # Helper to calculate expression weights (0.0 to 1.0)
        def _calculate_face_metrics(face_lms):
            if not face_lms: return [0.0, 0.0, 0.0, 0.0]
            
            # Key Landmarks for Expressions
            # Lips: 61(Left Corner), 291(Right Corner), 0(Top), 17(Bottom)
            # Brows: 65(Left Inner), 295(Right Inner)
            # Eyelids: 159(Left Top), 145(Left Bottom)
            # Face Frame: 234(Left Ear), 454(Right Ear), 10(Top Head), 152(Chin)
            
            try:
                lms = face_lms.landmark
                def dist(i1, i2): 
                    return np.sqrt((lms[i1].x - lms[i2].x)**2 + (lms[i1].y - lms[i2].y)**2)
                
                # Normalizers
                face_width = dist(234, 454)
                face_height = dist(10, 152) 
                
                if face_width < 0.01: return [0.0]*4

                # 1. Happy (Smile) - Ratio of mouth width to face width
                mouth_width = dist(61, 291)
                smile_ratio = (mouth_width / face_width)
                happy = max(0.0, min(1.0, (smile_ratio - 0.42) * 8.0)) # More sensitive

                # 2. Surprised (Mouth Open)
                mouth_open = dist(13, 14)
                open_ratio = (mouth_open / face_height)
                surprised = max(0.0, min(1.0, (open_ratio - 0.03) * 15.0)) 

                # 3. Angry (Brow Furrow)
                brow_dist = dist(55, 285) 
                brow_ratio = (brow_dist / face_width)
                angry = max(0.0, min(1.0, (0.24 - brow_ratio) * 20.0))  
                
                # 4. Blink (Eye Openness)
                left_eye_open = dist(159, 145) / face_height
                blink = 1.0 if left_eye_open < 0.02 else 0.0 
                
                return [happy, surprised, angry, blink]
            except:
                return [0.0, 0.0, 0.0, 0.0]

    def extract_frame_features(self, results):
        """Standardizes feature extraction: Pose(99) + Hands(126) + Face(4) = 229 features"""
        ref_x, ref_y = 0.5, 0.5
        if results.pose_landmarks:
            nose = results.pose_landmarks.landmark[0]
            ref_x, ref_y = nose.x, nose.y
        
        def get_coords(res_attr, num_pts=21):
            if not res_attr: return [0.0] * (num_pts * 3)
            pts = res_attr.landmark[:num_pts]
            return [c for lm in pts for c in [lm.x - ref_x, lm.y - ref_y, lm.z]]

        frame_features = []
        frame_features.extend(get_coords(results.left_hand_landmarks, 21))  # 63
        frame_features.extend(get_coords(results.right_hand_landmarks, 21)) # 63
        frame_features.extend(get_coords(results.pose_landmarks, 33))       # 99
        
        # Calculate expressions (4)
        def _get_metrics(face_landmarks):
            if not face_landmarks: return [0.0]*4
            try:
                def dist(a, b):
                    p1 = face_landmarks.landmark[a]
                    p2 = face_landmarks.landmark[b]
                    return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2)**0.5
                
                # Face metrics logic
                f_w = dist(234, 454)
                f_h = dist(10, 152)
                if f_w < 0.01: return [0.0]*4
                happy = max(0.0, min(1.0, (dist(61, 291)/f_w - 0.42) * 8.0))
                surp = max(0.0, min(1.0, (dist(13, 14)/f_h - 0.03) * 15.0))
                angry = max(0.0, min(1.0, (0.24 - dist(55, 285)/f_w) * 20.0))
                blink = 1.0 if (dist(159, 145)/f_h) < 0.02 else 0.0
                return [happy, surp, angry, blink]
            except: return [0.0]*4

        frame_features.extend(_get_metrics(results.face_landmarks))
        return frame_features

    def extract_landmarks_from_video(self, video_path, return_sequence=False, max_frames=60):
        cap = cv2.VideoCapture(str(video_path))
        features_sequence = []
        
        with self.mp_holistic.Holistic(
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5,
            refine_face_landmarks=True 
        ) as holistic:
            count = 0
            while cap.isOpened() and count < max_frames:
                ret, frame = cap.read()
                if not ret: break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb)
                features = self.extract_frame_features(results)
                features_sequence.append(features)
                count += 1
        
        cap.release()
        if not features_sequence: return None
        return np.array(features_sequence) if return_sequence else np.mean(features_sequence, axis=0)

    def build_landmark_dictionary(self, translator):
        """
        Synchronizes internal vocabulary with external benchmark assets.
        Generates and persists landmark DNA for all supported tokens.
        """
        for word, urdu in self.vocabulary.items():
            temp_v = self.videos_dir / f"{word}.mp4"
            if not temp_v.exists():
                try:
                    clip = translator.translate(urdu)
                    clip.save(str(temp_v), overwrite=True)
                except:
                    continue
            
            sequence = self.extract_landmarks_from_video(temp_v, return_sequence=True)
            if sequence is not None:
                self.landmark_dict[word] = sequence
                np.save(self.landmarks_dir / f"{word}.npy", sequence)

    def train_core(self):
        """
        Trains the classification matrix using skeletal DNA.
        Implements high-fidelity data augmentation for single-benchmark stability.
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
        
        if not self.landmark_dict:
            # Try loading from disk
            for npy in self.landmarks_dir.glob("*.npy"):
                word = npy.stem
                self.landmark_dict[word] = np.load(npy)
        
        if not self.landmark_dict: return False
        
        # --- BEST PRACTICE: DATA AUGMENTATION ---
        # With only 1 video per word, we must create synthetic variants
        X_aug = []
        y_aug = []
        
        for word, seq in self.landmark_dict.items():
            mean_vec = np.mean(seq, axis=0)
            # 1. Original
            X_aug.append(mean_vec)
            y_aug.append(word)
            
            # 2. Add variants (Noise + Scaling)
            for _ in range(50): # Create 50 variations per word
                noise = np.random.normal(0, 0.015, mean_vec.shape)
                scale = np.random.uniform(0.95, 1.05)
                variant = (mean_vec + noise) * scale
                X_aug.append(variant)
                y_aug.append(word)
        
        X = np.array(X_aug)
        y = np.array(y_aug)
        
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        self.classifier = RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=5, random_state=42)
        self.classifier.fit(X, y_encoded)
        
        # Save Model
        with open(self.model_path, 'wb') as f:
            pickle.dump((self.classifier, self.label_encoder), f)
        return True

    def load_core(self):
        """Load the trained model"""
        if self.model_path.exists():
            with open(self.model_path, 'rb') as f:
                self.classifier, self.label_encoder = pickle.load(f)
            return True
        return False

    def predict_sign(self, video_path):
        """Predict sign from video using CLR Core"""
        if not self.classifier: 
            print("⚠️ Classifier not loaded.")
            return None, 0
        landmarks = self.extract_landmarks_from_video(video_path)
        if landmarks is None or len(landmarks) == 0:
            print(f"⚠️ No landmarks detected in: {video_path}")
            return None, 0
        return self.predict_from_landmarks(landmarks)

    def predict_from_landmarks(self, landmarks_vector):
        """Predict sign directly from a landmark vector (Real-time Best Practice)"""
        if self.classifier is None or landmarks_vector is None:
            return None, 0
            
        # Landmarks arrive as a temporal sequence (frames, features)
        # We must average them to match the training vector (1, features)
        if len(landmarks_vector.shape) > 1:
            features = np.mean(landmarks_vector, axis=0).reshape(1, -1)
        else:
            features = landmarks_vector.reshape(1, -1)
        
        prob = self.classifier.predict_proba(features)[0]
        max_idx = np.argmax(prob)
        label = self.label_encoder.inverse_transform([max_idx])[0]
        return label, prob[max_idx] * 100

    def predict_sentence(self, video_path, energy_threshold=0.035, silence_padding=5):
        """
        Recognition Engine v3.0: Temporal Segmentation for Sentence Analysis.
        Splits a single video into multiple signs based on motion energy valleys.
        """
        if not self.classifier: return ["⚠️ Model not loaded"], 0
        
        # 1. Full Sequence Extraction
        full_sequence = self.extract_landmarks_from_video(video_path, max_frames=200)
        if full_sequence is None or len(full_sequence) < 10:
            return None, 0

        # 2. Motion Energy Calculation (Euclidean distance between adjacent frames)
        # We focus on hand landmarks (features 0-126) for most energy
        hand_motion = np.diff(full_sequence[:, :126], axis=0)
        energy = np.linalg.norm(hand_motion, axis=1)
        
        # 3. Smoothing (Moving Average)
        smooth_energy = np.convolve(energy, np.ones(5)/5, mode='same')
        
        # 4. Valley Detection (Segmentation)
        is_moving = smooth_energy > energy_threshold
        
        segments = []
        in_segment = False
        start_f = 0
        
        for i, moving in enumerate(is_moving):
            if moving and not in_segment:
                start_f = max(0, i - silence_padding)
                in_segment = True
            elif not moving and in_segment:
                end_f = min(len(full_sequence), i + silence_padding)
                if (end_f - start_f) > 8: # Minimum 8 frames per word
                    segments.append(full_sequence[start_f:end_f])
                in_segment = False
        
        # Catch last segment
        if in_segment:
            segments.append(full_sequence[start_f:])

        # 5. Iterative Classification
        results = []
        total_conf = 0
        for seg in segments:
            label, conf = self.predict_from_landmarks(seg)
            if label and conf > 45: # Filter noise
                results.append(label)
                total_conf += conf
        
        if not results: return None, 0
        
        avg_conf = total_conf / len(results)
        return results, avg_conf

    def get_word_dna(self, word):
        """Retrieve the Skeletal DNA (landmarks) for a specific word"""
        word = word.lower()
        if word in self.landmark_dict:
            return self.landmark_dict[word]
        
        # Try loading from disk
        path = self.landmarks_dir / f"{word}.npy"
        if path.exists():
            return np.load(path)
        return None

    def get_available_words(self):
        """Returns list of words that have valid landmark files (i.e., can be used)."""
        available = []
        for word in self.vocabulary.keys():
            path = self.landmarks_dir / f"{word}.npy"
            if path.exists():
                available.append(word)
        return available

    def export_dna_json(self, dna_sequence):
        """Converts a numpy DNA sequence into a JSON-friendly list of dictionaries for 3D rigging."""
        if dna_sequence is None: return None
        json_sequence = []
        for frame in dna_sequence:
            # Hand landmarks are 21 pts * 3 = 63 features each
            # Pose landmarks are 33 pts * 3 = 99 features
            # Expressions (if present) are the remaining features
            frame_data = {
                "left_hand": frame[0:63].tolist(),
                "right_hand": frame[63:126].tolist(),
                "pose": frame[126:225].tolist()
            }
            
            # Dynamic Expression Export
            if len(frame) > 225:
                # Map available expression floats (happy, surprised, angry, blink)
                exprs = frame[225:]
                frame_data["expressions"] = {
                    "happy": float(exprs[0]) if len(exprs) > 0 else 0.0,
                    "surprised": float(exprs[1]) if len(exprs) > 1 else 0.0,
                    "angry": float(exprs[2]) if len(exprs) > 2 else 0.0,
                    "blink": float(exprs[3]) if len(exprs) > 3 else 0.0
                }
            else:
                frame_data["expressions"] = {"happy": 0.0, "surprised": 0.0, "angry": 0.0, "blink": 0.0}
            
            json_sequence.append(frame_data)
        return json_sequence

    def get_words_dna_json(self, words_list):
        """Returns the full stitched DNA for a sentence in JSON-ready format."""
        dna_sequences = []
        for w in words_list:
            dna = self.get_word_dna(w)
            if dna is not None:
                dna_sequences.append(dna)
        
        if not dna_sequences: return None
        
        # Use existing instance if possible or create a localized one safely
        renderer = DigitalHumanRenderer()
        full_seq = renderer.stitch_landmarks(dna_sequences)
        return self.export_dna_json(full_seq)

    def speech_to_text(self):
        """Convert live speech to text for translation input (Graceful handles server lack of mic)"""
        try:
            import speech_recognition as sr
            r = sr.Recognizer()
            with sr.Microphone() as source:
                print("🎤 Listening...")
                audio = r.listen(source, timeout=5, phrase_time_limit=10)
                text = r.recognize_google(audio)
                return text
        except (OSError, ImportError) as e:
            print(f"🎙️ Mic/Speech Error: {e}")
            return "ERROR: Microphone not available on server. Please type your text."
        except Exception as e:
            print(f"🎙️ Speech Error: {e}")
            return None

class DigitalHumanRenderer:
    """
    Synthesizes a high-fidelity Digital Human Avatar from skeletal landmarks.
    This is the 'Best Practice' for generating clean, focused sign language output.
    """
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic
        
        # Professional Styling (Cyan Neon Theme)
        self.node_spec = self.mp_drawing.DrawingSpec(color=(248, 189, 56), thickness=2, circle_radius=2)
        self.link_spec = self.mp_drawing.DrawingSpec(color=(56, 189, 248), thickness=3)
        
        # User Photo Integration
        self.user_face = self._extract_user_face()

    def _extract_user_face(self):
        """Extracts and crops the user's face from user_profile.png if available"""
        path = os.path.join(os.path.dirname(__file__), "user_profile.png")
        if not os.path.exists(path): return None
        try:
            img = cv2.imread(path)
            if img is None: return None
            
            # Intelligent Center Crop (assuming the head is in the top-middle)
            h, w = img.shape[:2]
            # We take a square crop based on the smaller dimension
            size = min(h, w)
            # Offset center slightly up for profile photos (usually head is top-ish)
            start_x = (w - size) // 2
            start_y = max(0, (h - size) // 2 - int(h * 0.1)) # Shift up by 10%
            
            crop = img[start_y:start_y+size, start_x:start_x+size]
            radius = size // 2
            
            # Create Circular Mask
            mask = np.zeros((size, size), dtype=np.uint8)
            cv2.circle(mask, (radius, radius), radius, 255, -1)
            
            # Add Alpha Channel
            face_rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
            face_rgba[:, :, 3] = mask
            return cv2.resize(face_rgba, (140, 140)) # Slightly larger for better detail
        except Exception as e:
            print(f"⚠️ Face Extraction Failed: {e}")
            return None

    def render_landmark_dna(self, landmark_sequence, output_path, width=640, height=480, fps=30):
        """Renders raw landmarks into a stylized digital human video clip (Skeletal Mode)"""
        if landmark_sequence is None or len(landmark_sequence) == 0: return None
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        for frame_vec in landmark_sequence:
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            canvas[:] = (15, 23, 42) # Slate-900 Background
            cx, cy = width // 2, height // 2
            def draw_points(points, start_idx, num_points, color):
                for i in range(num_points):
                    idx = start_idx + (i * 3)
                    if idx + 2 >= len(points): break
                    px = int(cx + (points[idx] * width * 0.8))
                    py = int(cy + (points[idx+1] * height * 0.8))
                    cv2.circle(canvas, (px, py), 4, color, -1)
                    cv2.circle(canvas, (px, py), 2, (255, 255, 255), -1)
            draw_points(frame_vec, 0, 21, (56, 189, 248)) # Hands
            draw_points(frame_vec, 63, 21, (56, 189, 248)) 
            draw_points(frame_vec, 126, 33, (34, 197, 94)) # Pose
            cv2.putText(canvas, "MODE: SKELETAL BENCHMARK", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (56, 189, 248), 1)
            out.write(canvas)
        out.release()
        return output_path

    def render_neo_avatar(self, landmark_sequence, output_path, width=640, height=480, fps=30):
        """Renders landmarks into a premium 3D-style Neo-Avatar (Animate Mode)"""
        if landmark_sequence is None or len(landmark_sequence) == 0: return None
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame_vec in landmark_sequence:
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            canvas[:] = (30, 41, 59) # Slate-800 Background
            cx, cy = width // 2, height // 2
            
            def get_p(idx):
                i = idx * 3
                return (int(cx + (frame_vec[i] * width * 0.8)), int(cy + (frame_vec[i+1] * height * 0.8)))

            # 1. Draw Volumetric Torso (Corporate Suit)
            try:
                p_l_sh = get_p(11 + 42) 
                p_r_sh = get_p(12 + 42)
                p_l_hip = get_p(23 + 42)
                p_r_hip = get_p(24 + 42)
                
                # Suit Jacket
                poly = np.array([p_l_sh, p_r_sh, p_r_hip, p_l_hip], np.int32)
                cv2.fillPoly(canvas, [poly], (15, 23, 42)) # Dark Navy Suit
                cv2.polylines(canvas, [poly], True, (51, 65, 85), 2)
                
                # Shirt/Tie Area
                neck_center = ((p_l_sh[0] + p_r_sh[0])//2, (p_l_sh[1] + p_r_sh[1])//2)
                chest_center = ((p_l_hip[0] + p_r_hip[0])//2, (p_l_hip[1] + p_r_hip[1])//2)
                shirt_poly = np.array([p_l_sh, neck_center, p_r_sh, chest_center], np.int32) # Simple V-neck
                cv2.fillPoly(canvas, [shirt_poly], (241, 245, 249)) # White Shirt
                
                # Tie (Konecta Blue)
                tie_tip = (chest_center[0], int(neck_center[1] + (chest_center[1] - neck_center[1])*0.6))
                tie_poly = np.array([neck_center, (neck_center[0]-5, neck_center[1]+10), tie_tip, (neck_center[0]+5, neck_center[1]+10)], np.int32)
                cv2.fillPoly(canvas, [tie_poly], (56, 189, 248)) 
                
                # Head (If user photo exists, use it! Otherwise use Geometric)
                p_nose = get_p(0 + 42)
                if self.user_face is not None:
                    # Overlay User Photo
                    fw, fh = self.user_face.shape[1], self.user_face.shape[0]
                    y1, y2 = p_nose[1]-fh//2, p_nose[1]+fh//2
                    x1, x2 = p_nose[0]-fw//2, p_nose[0]+fw//2
                    if y1 >= 0 and y2 < height and x1 >= 0 and x2 < width:
                        overlay = self.user_face[:, :, :3]
                        mask = self.user_face[:, :, 3] / 255.0
                        for c in range(3):
                            canvas[y1:y2, x1:x2, c] = (1 - mask) * canvas[y1:y2, x1:x2, c] + mask * overlay[:, :, c]
                    cv2.circle(canvas, p_nose, fh//2, (241, 245, 249), 2) # Border
                else:
                    # Professional Geometric Face
                    cv2.circle(canvas, p_nose, 40, (226, 232, 240), -1) 
                    cv2.circle(canvas, p_nose, 40, (241, 245, 249), 2)

                # Neon Facial Tracking Sync (Eyes & Mouth - Always visible for tracking feedback)
                try:
                    p_l_eye = get_p(2 + 42)
                    p_r_eye = get_p(5 + 42)
                    # Glow Eyes
                    cv2.circle(canvas, p_l_eye, 4, (56, 189, 248), -1)
                    cv2.circle(canvas, p_l_eye, 2, (255, 255, 255), -1)
                    cv2.circle(canvas, p_r_eye, 4, (56, 189, 248), -1)
                    cv2.circle(canvas, p_r_eye, 2, (255, 255, 255), -1)
                    
                    # Expressive Mouth
                    p_l_mouth = get_p(9 + 42)
                    p_r_mouth = get_p(10 + 42)
                    cv2.line(canvas, p_l_mouth, p_r_mouth, (255, 255, 255), 2)
                except: pass
            except: pass

            # 2. Draw Limbs
            def draw_limb(i1, i2, color, thickness=12):
                try: cv2.line(canvas, get_p(i1+42), get_p(i2+42), color, thickness)
                except: pass

            draw_limb(11, 13, (15, 23, 42), 18) # Left Upper Arm (Sleeve)
            draw_limb(13, 15, (226, 232, 240), 12) # forearm (Skin)
            draw_limb(12, 14, (15, 23, 42), 18) # Right Upper Arm
            draw_limb(14, 16, (226, 232, 240), 12)

            # 3. Draw Hands (Biological Refinement: Skeletal + Solid Palm)
            def draw_hand(start_idx, color):
                # Correct MediaPipe Joint Mapping
                finger_connections = [
                    (0,1), (1,2), (2,3), (3,4), # Thumb
                    (0,5), (5,6), (6,7), (7,8), # Index
                    (0,9), (9,10), (10,11), (11,12), # Middle
                    (0,13), (13,14), (14,15), (15,16), # Ring
                    (0,17), (17,18), (18,19), (19,20) # Pinky
                ]
                
                # Solid Palm Construction
                palm_indices = [0, 1, 5, 9, 13, 17, 0]
                palm_points = []
                for i in palm_indices:
                    idx = (start_idx + i) * 3
                    palm_points.append([int(cx + (frame_vec[idx] * width * 0.8)), int(cy + (frame_vec[idx+1] * height * 0.8))])
                
                # Draw Anatomical Palm Base
                cv2.fillPoly(canvas, [np.array(palm_points, np.int32)], color, cv2.LINE_AA)
                
                # Draw Connected Bone Lines
                for i1, i2 in finger_connections:
                    try:
                        idx1, idx2 = (start_idx + i1) * 3, (start_idx + i2) * 3
                        p1 = (int(cx + (frame_vec[idx1] * width * 0.8)), int(cy + (frame_vec[idx1+1] * height * 0.8)))
                        p2 = (int(cx + (frame_vec[idx2] * width * 0.8)), int(cy + (frame_vec[idx2+1] * height * 0.8)))
                        cv2.line(canvas, p1, p2, color, 4, cv2.LINE_AA)
                    except: pass
                
                # Round Joint Junctions
                for i in range(21):
                    idx = (start_idx + i) * 3
                    px, py = int(cx + (frame_vec[idx] * width * 0.8)), int(cy + (frame_vec[idx+1] * height * 0.8))
                    cv2.circle(canvas, (px, py), 4, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(canvas, (px, py), 2, color, -1, cv2.LINE_AA)
            
            draw_hand(0, (56, 189, 248))  # Left Hand (Neo Blue)
            draw_hand(21, (244, 114, 182)) # Right Hand (Neo Pink)
            
            # 4. Connected Limbs Refinement (Arm-to-Hand)
            try:
                draw_limb(15, 0, (226, 232, 240), 10) # Left wrist connection
                draw_limb(16, 21, (226, 232, 240), 10) # Right wrist connection
            except: pass

            cv2.putText(canvas, "KONECTA AI REPRESENTATIVE v1.0", (width-320, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (56, 189, 248), 1)
            out.write(canvas)
            
        out.release()
        return output_path

    def stitch_landmarks(self, dna_list, transition_frames=10):
        if not dna_list: return None
        stitched_sequence = [np.array([dna_list[0][0]] * 5)]
        stitched_sequence.append(dna_list[0])
        for i in range(1, len(dna_list)):
            prev_seq, curr_seq = dna_list[i-1], dna_list[i]
            last_f, first_f = prev_seq[-1], curr_seq[0]
            interp = []
            for t in range(1, transition_frames + 1):
                alpha = t / (transition_frames + 1)
                interp.append((1 - alpha) * last_f + alpha * first_f)
            stitched_sequence.append(np.array(interp))
            stitched_sequence.append(curr_seq)
        stitched_sequence.append(np.array([dna_list[-1][-1]] * 5))
        return np.concatenate(stitched_sequence, axis=0)

    def stitch_and_render(self, dna_list, output_path, transition_frames=10):
        full_seq = self.stitch_landmarks(dna_list, transition_frames)
        return self.render_landmark_dna(full_seq, output_path)
