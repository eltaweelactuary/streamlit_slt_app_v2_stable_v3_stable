import os
import time
import builtins
import tempfile
import streamlit as st
import shutil
import cv2

# ==============================================================================
# --- CRITICAL: GLOBAL MONKEYPATCH FOR STREAMLIT CLOUD PERMISSIONS ---
# ==============================================================================
WRITABLE_BASE = os.path.join(tempfile.gettempdir(), "slt_persistent_storage")
os.makedirs(WRITABLE_BASE, exist_ok=True)

_orig_makedirs = os.makedirs
_orig_mkdir = os.mkdir
_orig_open = builtins.open
_orig_rename = os.rename
_orig_replace = os.replace
_orig_exists = os.path.exists
_orig_isfile = os.path.isfile
_orig_listdir = os.listdir

def _get_shadow_path(path):
    if not path: return path
    p = str(path).replace("\\", "/")
    if "site-packages/sign_language_translator" in p:
        parts = p.split("site-packages/sign_language_translator/")
        rel = parts[1] if len(parts) > 1 else ""
        return os.path.join(WRITABLE_BASE, rel)
    return None

def _redirect_read_write(path, is_write=False):
    shadow = _get_shadow_path(path)
    if shadow:
        if is_write or _orig_exists(shadow):
            parent = os.path.dirname(shadow)
            if parent and not _orig_exists(parent):
                _orig_makedirs(parent, exist_ok=True)
            return shadow
    return path

def _patched_makedirs(name, mode=0o777, exist_ok=False):
    return _orig_makedirs(_redirect_read_write(name, is_write=True), mode, exist_ok)

def _patched_mkdir(path, mode=0o777, *args, **kwargs):
    return _orig_mkdir(_redirect_read_write(path, is_write=True), mode, *args, **kwargs)

def _patched_open(file, *args, **kwargs):
    mode = args[0] if args else kwargs.get('mode', 'r')
    is_write = any(m in mode for m in ('w', 'a', '+', 'x'))
    return _orig_open(_redirect_read_write(file, is_write=is_write), *args, **kwargs)

def _patched_rename(src, dst, *args, **kwargs):
    return _orig_rename(_redirect_read_write(src), _redirect_read_write(dst, is_write=True), *args, **kwargs)

def _patched_replace(src, dst, *args, **kwargs):
    return _orig_replace(_redirect_read_write(src), _redirect_read_write(dst, is_write=True), *args, **kwargs)

def _patched_exists(path):
    shadow = _get_shadow_path(path)
    if shadow and _orig_exists(shadow): return True
    return _orig_exists(path)

def _patched_isfile(path):
    shadow = _get_shadow_path(path)
    if shadow and _orig_isfile(shadow): return True
    return _orig_isfile(path)

def _patched_listdir(path):
    return _orig_listdir(_redirect_read_write(path))

# Apply global patches
os.makedirs = _patched_makedirs
os.mkdir = _patched_mkdir
builtins.open = _patched_open
os.rename = _patched_rename
os.replace = _patched_replace
os.path.exists = _patched_exists
os.path.isfile = _patched_isfile
os.listdir = _patched_listdir

# --- ENCODING FIX: Standard Linux/Streamlit lack H.264 encoder ---
import subprocess

def _optimize_video_for_web(path):
    """Converts a video to H.264 using ffmpeg to ensure browser compatibility."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return
    
    # We use a temp file for output
    temp_out = path + ".optimized.mp4"
    try:
        # ffmpeg command to convert to H.264 (libx264)
        # -y: overwrite
        # -preset ultrafast: for speed
        # -vcodec libx264: the standard for web
        cmd = [
            "ffmpeg", "-y", "-i", path, 
            "-vcodec", "libx264", 
            "-pix_fmt", "yuv420p", 
            "-preset", "ultrafast", 
            temp_out
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        if os.path.exists(temp_out) and os.path.getsize(temp_out) > 0:
            os.replace(temp_out, path)
            print(f"🎬 Video Optimized: {path}")
    except Exception as e:
        print(f"⚠️ Video Optimization Failed for {path}: {e}")
        if os.path.exists(temp_out): os.remove(temp_out)

class _VideoWriterWrapper:
    def __init__(self, writer, filename):
        self.writer = writer
        self.filename = filename
    def write(self, frame):
        self.writer.write(frame)
    def release(self):
        self.writer.release()
        _optimize_video_for_web(self.filename)
    def __getattr__(self, name):
        return getattr(self.writer, name)

_orig_VideoWriter = cv2.VideoWriter
def _patched_VideoWriter(filename, fourcc, *args, **kwargs):
    # Force mp4v for high-compatibility saving on Linux
    # Then we optimize it to H.264 on release
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    writer = _orig_VideoWriter(filename, fourcc, *args, **kwargs)
    return _VideoWriterWrapper(writer, filename)

cv2.VideoWriter = _patched_VideoWriter
# -----------------------------------------------------------------

# Initialize Assets ROOT_DIR immediately
try:
    import sign_language_translator as slt
    slt.Assets.ROOT_DIR = WRITABLE_BASE
except:
    pass
# ==============================================================================

import numpy as np
import pickle
from pathlib import Path
from sign_language_core import SignLanguageCore, DigitalHumanRenderer

# Page Config
st.set_page_config(
    page_title="Digital Human SLT",
    page_icon="🤖",
    layout="wide"
)

# Premium UI Styling
st.markdown("""
<style>
    .main { background-color: #0f172a; color: white; }
    .stApp { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); color: white; }
    h1, h2, h3 { color: #38bdf8 !important; }
    div[data-testid="stExpander"] { background-color: rgba(56, 189, 248, 0.05); border: 1px solid #38bdf8; border-radius: 10px; }
    .stTextInput>div>div>input { background-color: #1e293b; color: white; border: 1px solid #38bdf8; }
</style>
""", unsafe_allow_html=True)

# Vocabulary Mapping (Expanded for UI display)
PSL_VOCABULARY = {
    "apple": "سیب", "world": "دنیا", "pakistan": "پاکستان",
    "good": "اچھا", "red": "لال", "is": "ہے", "the": "یہ", "that": "وہ",
    "hello": "ہیلو", "salam": "سلام", "welcome": "خوش آمدید",
    "thank you": "شکریہ", "yes": "ہاں", "no": "نہیں", "please": "براہ کرم",
    "I": "میں", "you": "تم", "we": "ہم", "he": "وہ", "she": "وہ",
    "name": "نام", "my": "میرا", "your": "تمہارا",
    "eat": "کھانا", "drink": "پینا", "go": "جانا", "come": "آنا",
    "help": "مدد", "water": "پانی", "food": "کھانا",
    "house": "گھر", "school": "اسکول", "book": "کتاب",
    "happy": "خوش", "sad": "اداس", "angry": "غصہ",
    "what": "کیا", "where": "کہاں", "how": "کیسے"
}

# App Data Paths
DATA_DIR = os.path.join(WRITABLE_BASE, "app_internal_data")
os.makedirs(DATA_DIR, exist_ok=True)

@st.cache_resource
def get_slt_core():
    core = SignLanguageCore(data_dir=DATA_DIR)
    core.load_core()
    return core

@st.cache_resource
def get_avatar_renderer():
    return DigitalHumanRenderer()

@st.cache_resource
def load_slt_engine():
    import sign_language_translator as slt
    # Explicitly ensure the library uses our writable path (Shadow Persistence)
    slt.Assets.ROOT_DIR = WRITABLE_BASE
    # Sync core data directory to the same base to avoid redundant downloads
    core_data_dir = os.path.join(WRITABLE_BASE, "app_internal_data")
    os.makedirs(core_data_dir, exist_ok=True)
    
    translator = slt.models.ConcatenativeSynthesis(
        text_language="urdu",
        sign_language="psl",
        sign_format="vid"
    )
    return translator, slt

def load_or_train_core(core, translator):
    if core.classifier: return True
    st.info("🔧 Building Next-Gen Landmark Dictionary (First run)...")
    core.build_landmark_dictionary(translator)
    st.info("🧠 Training CLR Core Brain (v2 Matrix)...")
    if core.train_core():
        print("🚀 SLT Core Brain Training Complete!")
        st.success("✅ Core Model trained successfully.")
        return True
    return False

def main():
    # ==== FORCE CACHE CLEAR: Delete old landmarks to trigger re-extraction ====
    # This ensures the new Face Mesh expression logic is applied
    cache_path = os.path.join(WRITABLE_BASE, "app_internal_data", "landmarks")
    model_path = os.path.join(WRITABLE_BASE, "app_internal_data", "models")
    if os.path.exists(cache_path):
        import shutil
        shutil.rmtree(cache_path, ignore_errors=True)
        if os.path.exists(model_path):
            shutil.rmtree(model_path, ignore_errors=True)
        print("🗑️ CACHE CLEARED - Landmarks & Model will rebuild on this run")
    # ============================================================================
    
    st.title("🤟 Sign Language Translator")
    st.markdown("**Bidirectional Translation:** Text ↔ Pakistani Sign Language (PSL)")
    st.markdown("---")
    
    # Architecture Explanation
    with st.expander("📚 System Architecture: Unified Data Representation"):
        st.markdown("""
        The system relies on a **Common Landmark Benchmark**:
        1. **Text → Video:** Maps text to the Benchmark Dictionary.
        2. **Video → Text:** Extracts landmarks and compares them against the same Benchmark.
        """)
    
    with st.spinner("⏳ Loading SLT Core & Avatar Engine..."):
        translator, slt = load_slt_engine()
        core = get_slt_core()
        renderer = get_avatar_renderer()
    
    if not load_or_train_core(core, translator):
        st.error("❌ Failed to initialize SLT Core.")
        st.stop()
    
    st.success(f"✅ System Ready | Vocabulary Size: {len(core.landmark_dict if core.landmark_dict else [])}")
    
    tab1, tab2 = st.tabs(["📝 Text → Video", "🎥 Video → Text"])
    
    # TAB 1: TEXT TO VIDEO
    with tab1:
        st.header("📝 Text to Sign Language Video")
        st.info(f"**Available words:** {', '.join(PSL_VOCABULARY.keys())}")
        
        text_input = st.text_input("Enter text:", placeholder="e.g., apple good world")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            gen_btn = st.button("🚀 Generate Digital Human Output")
        with col2:
            if st.button("🎤 Use Voice Input"):
                with st.spinner("🎙️ Listening..."):
                    voice_text = core.speech_to_text()
                    if voice_text:
                        st.info(f"🎤 Heard: **{voice_text}**")
                        text_input = voice_text

        if gen_btn and text_input:
            with st.spinner("🧪 Transforming to Digital Avatar..."):
                words = text_input.lower().split()
                st.session_state['last_words'] = words
                v_clips = []
                dna_list = []
                
                status_placeholder = st.empty() # Added for dynamic status updates
                for w in words:
                    if w in PSL_VOCABULARY:
                        try:
                            status_placeholder.info(f"🔍 Processing: **{w}**") # Changed to use placeholder
                            clip = translator.translate(PSL_VOCABULARY[w])
                            v_clips.append(clip)
                            dna = core.get_word_dna(w)
                            if dna is not None:
                                dna_list.append(dna)
                                status_placeholder.success(f"✅ DNA for **{w}** ready.") # Changed to use placeholder
                            else:
                                status_placeholder.warning(f"⚠️ DNA for **{w}** missing in dictionary.") # Changed to use placeholder
                        except Exception as e:
                            status_placeholder.error(f"❌ Error synthesizing **{w}**: {e}") # Changed to use placeholder
                    else:
                        status_placeholder.warning(f"📖 Word **{w}** not in vocabulary.") # Changed to use placeholder
                
                if v_clips:
                    # 1. Standard Benchmark (Stitched)
                    f_orig = v_clips[0]
                    for c in v_clips[1:]: f_orig = f_orig + c
                    p_benchmark = os.path.join(tempfile.gettempdir(), f"bench_{int(time.time())}.mp4")
                    f_orig.save(p_benchmark, overwrite=True)
                    _optimize_video_for_web(p_benchmark)
                    
                    # 2. Skeletal & Neo-Avatar DNA
                    full_dna = renderer.stitch_landmarks(dna_list)
                    
                    p_skeletal = os.path.join(tempfile.gettempdir(), f"skel_{int(time.time())}.mp4")
                    renderer.render_landmark_dna(full_dna, p_skeletal)
                    _optimize_video_for_web(p_skeletal)
                    
                    p_neo = os.path.join(tempfile.gettempdir(), f"neo_{int(time.time())}.mp4")
                    renderer.render_neo_avatar(full_dna, p_neo)
                    _optimize_video_for_web(p_neo)
                    
                    # Store everything in session state (Strings/Numeric only)
                    st.session_state['benchmark_path'] = p_benchmark
                    st.session_state['skeletal_path'] = p_skeletal
                    st.session_state['neo_path'] = p_neo
                    st.session_state['full_dna'] = full_dna
                    st.session_state['last_words'] = words
                    
                    # Explicit memory cleanup
                    del v_clips
                    del full_dna
                    import gc
                    gc.collect()

        # PERSISTENT DISPLAY SECTION (Works even after toggle)
        if 'benchmark_path' in st.session_state:
            st.divider()
            cinema_mode = st.toggle("🎭 Activate Cinema Mode (3D VRM Client)", value=False)
            
            words = st.session_state.get('last_words', [])
            full_dna = st.session_state.get('full_dna')
            
            if cinema_mode:
                st.markdown("### 🎬 Cinema Mode: High-Fidelity 3D Avatar")
                dna_json = core.get_words_dna_json(words)
                if dna_json:
                    import json
                    import base64
                    
                    # Embed local VRM model for stability
                    vrm_path = "5084648674725325209.vrm" # New VRM with BlendShapes
                    vrm_base64 = ""
                    if os.path.exists(vrm_path):
                        with open(vrm_path, "rb") as f:
                            vrm_base64 = base64.b64encode(f.read()).decode()
                    else:
                        st.error("❌ VRM Model file not found. Please upload 'VRM1_Constraint_Twist_Sample.vrm' to the app directory.")
                    
                    st.info("🤖 **Local VRM Bridge Active** | Transmitting Skeletal DNA...")
                    
                    html_component = f"""
                    <!DOCTYPE html>
                    <html><head>
                    <style>
                        * {{ margin:0; padding:0; box-sizing:border-box; }}
                        body {{ background: linear-gradient(135deg, #1a1a2e, #16213e); display:flex; justify-content:center; align-items:center; min-height:100vh; font-family:'Inter',sans-serif; color:#fff; }}
                        #canvas {{ width:100%; height:450px; border-radius:12px; }}
                        .hud {{ display:flex; justify-content:space-around; margin-top:10px; padding:10px; background:rgba(255,255,255,0.05); border-radius:8px; }}
                        .hud-item {{ text-align:center; }}
                        .hud-label {{ font-size:0.7rem; color:#aaa; text-transform:uppercase; }}
                        .hud-value {{ font-size:1.1rem; font-weight:bold; color:#0f9d58; }}
                    </style>
                    </head><body>
                    <div style="width:100%; max-width:700px; padding:10px;">
                        <canvas id="canvas"></canvas>
                        <div class="hud">
                            <div class="hud-item"><div class="hud-label">Status</div><div class="hud-value" id="status">Loading...</div></div>
                            <div class="hud-item"><div class="hud-label">Frame</div><div class="hud-value" id="frame">0/{len(dna_json)}</div></div>
                            <div class="hud-item"><div class="hud-label">Words</div><div class="hud-value">{' '.join(words)}</div></div>
                        </div>
                    </div>
                    <script type="importmap">
                    {{
                        "imports": {{
                            "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
                            "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
                        }}
                    }}
                    </script>
                    <script type="module">
                        import * as THREE from 'three';
                        import {{ GLTFLoader }} from 'three/addons/loaders/GLTFLoader.js';
                        import {{ VRMLoaderPlugin, VRMUtils }} from 'https://cdn.jsdelivr.net/npm/@pixiv/three-vrm@2.0.6/lib/three-vrm.module.js';

                        const DNA = {json.dumps(dna_json)};
                        let vrm, frameIdx = 0;
                        const canvas = document.getElementById('canvas');
                        const scene = new THREE.Scene();
                        scene.background = new THREE.Color(0x1a1a2e);
                        const camera = new THREE.PerspectiveCamera(35, canvas.clientWidth / canvas.clientHeight, 0.1, 100);
                        camera.position.set(0, 1.0, 3);
                        camera.lookAt(0, 1, 0);
                        const renderer = new THREE.WebGLRenderer({{ canvas, antialias: true }});
                        renderer.setSize(canvas.clientWidth, canvas.clientHeight);
                        renderer.setPixelRatio(window.devicePixelRatio);
                        scene.add(new THREE.AmbientLight(0xffffff, 0.6));
                        const dirLight = new THREE.DirectionalLight(0xffffff, 1.5);
                        dirLight.position.set(1, 2, 1).normalize();
                        scene.add(dirLight);

                        const loader = new GLTFLoader();
                        loader.register(p => new VRMLoaderPlugin(p));
                        
                        // Using Local Base64 Model for Ultimate Stability
                        const vrmData = "data:application/octet-stream;base64," + "{vrm_base64}";
                        
                        if ("{vrm_base64}") {{
                            loader.load(vrmData, gltf => {{
                                vrm = gltf.userData.vrm;
                                VRMUtils.rotateVRM0(vrm);
                                scene.add(vrm.scene);
                                document.getElementById('status').textContent = 'Playing (Local)';
                            }}, null, e => {{ 
                                document.getElementById('status').textContent = 'Load Error: Corrupted Data';
                                console.error(e);
                            }});
                        }} else {{
                             document.getElementById('status').textContent = 'Load Error: No Model Data';
                        }}

                        const clock = new THREE.Clock();
                        
                        // --- ADVANCED FK SOLVER (Vector-to-Quaternion) ---
                        function solveBoneRotation(node, p_start, p_end, baseDir) {{
                            if (!node || !p_start || !p_end) return;
                            
                            const vTarget = new THREE.Vector3(
                                p_end[0] - p_start[0],
                                -(p_end[1] - p_start[1]), 
                                -(p_end[2] - p_start[2])
                            ).normalize();
                            
                            const q = new THREE.Quaternion().setFromUnitVectors(baseDir, vTarget);
                            
                            // SMOOTHING: Low alpha (0.1) creates "Heavy/Cinematic" feel, filtering jitter
                            node.quaternion.slerp(q, 0.1); 
                        }}
                        
                        // --- SIMPLIFIED HAND SOLVER: Direction-Based (matches VRM bone hierarchy) ---
                        function solveHandOrientation(node, p_wrist, p_index, p_middle, p_pinky, side) {{
                           if (!node || !p_wrist || !p_middle) return;
                           
                           // Convert to VRM coordinate space (flip Y and Z)
                           const toVec3 = (a) => new THREE.Vector3(a[0], -a[1], -a[2]);
                           
                           const vWrist = toVec3(p_wrist);
                           const vMiddle = toVec3(p_middle);

                           // Hand Direction: Wrist -> Middle finger base
                           const vDir = new THREE.Vector3().subVectors(vMiddle, vWrist);
                           
                           // Stability Check
                           if (vDir.lengthSq() < 0.0001) return;
                           vDir.normalize();

                           // VRM T-Pose: Left hand points +X, Right hand points -X
                           const defaultDir = new THREE.Vector3(side === 'left' ? 1 : -1, 0, 0);
                           
                           // Calculate rotation from default to target
                           const q = new THREE.Quaternion().setFromUnitVectors(defaultDir, vDir);

                           // Apply smoothly
                           node.quaternion.slerp(q, 0.2);
                        }}
                        
                        // --- NEUTRAL POSE (REST) ---
                        function resetArmToRest(arm, forearm, side) {{
                            if (!arm || !forearm) return;
                            // Strict Attention: Arm straight down
                            // Left Side: 1.5 radians (~85 deg)
                            const qArmDown = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, 0, side === 'left' ? 1.5 : -1.5)); 
                            
                            // Force Snap if Left Hand (Attention Pose)
                            const speed = (side === 'left') ? 0.2 : 0.1;
                            
                            arm.quaternion.slerp(qArmDown, speed);
                            forearm.quaternion.slerp(new THREE.Quaternion(), speed); 
                        }}

                        const fps = 30;
                        const timePerFrame = 1.0 / fps;
                        let timeAccumulator = 0.0;

                        function animate() {{
                            requestAnimationFrame(animate);
                            
                            const delta = clock.getDelta();
                            timeAccumulator += delta;
                            
                            if (vrm) {{
                                vrm.update(delta); // Update physics/blink logic at high freq
                                
                                // Lock Pose Updates to 30 FPS to match Video Speed
                                if (DNA.length > 0 && timeAccumulator >= timePerFrame) {{
                                    timeAccumulator %= timePerFrame; // Reset accumulator safely
                                    
                                    const frame = DNA[frameIdx];
                                    if (frame) {{
                                        
                                        // --- FACIAL EXPRESSIONS ---
                                        let debugExpr = "😐";
                                        if (frame.expressions && vrm.expressionManager) {{
                                            const expr = frame.expressions;
                                            vrm.expressionManager.setValue('happy', expr.happy || 0);
                                            vrm.expressionManager.setValue('surprised', expr.surprised || 0);
                                            vrm.expressionManager.setValue('angry', expr.angry || 0);
                                            vrm.expressionManager.setValue('blink', expr.blink || 0);
                                            
                                            // Debug Text
                                            if (expr.happy > 0.3) debugExpr = "😊";
                                            else if (expr.angry > 0.3) debugExpr = "😠";
                                            else if (expr.surprised > 0.3) debugExpr = "😲";
                                            
                                            // Update HUD
                                            const hudExpr = document.getElementById('status');
                                            if (hudExpr) hudExpr.textContent = 'Playing ' + debugExpr;
                                        }}

                                        if (frame.pose) {{
                                            const pose = frame.pose;
                                            const getPoseLM = (idx) => (pose.length > idx*3) ? [pose[idx*3], pose[idx*3+1], pose[idx*3+2]] : [0,0,0];
                                            const getLeftHandLM = (idx) => (frame.left_hand && frame.left_hand.length > idx*3) ? [frame.left_hand[idx*3], frame.left_hand[idx*3+1], frame.left_hand[idx*3+2]] : null;
                                            const getRightHandLM = (idx) => (frame.right_hand && frame.right_hand.length > idx*3) ? [frame.right_hand[idx*3], frame.right_hand[idx*3+1], frame.right_hand[idx*3+2]] : null;

                                            // --- RIGGING LOGIC ---
                                            
                                            // 1. Left Arm Chain (VRM: +X)
                                            const leftArm = vrm.humanoid.getNormalizedBoneNode('leftUpperArm');
                                            const leftForeArm = vrm.humanoid.getNormalizedBoneNode('leftLowerArm');
                                            const leftHand = vrm.humanoid.getNormalizedBoneNode('leftHand');
                                            
                                            const lShoulder = getPoseLM(11);
                                            const lElbow = getPoseLM(13);
                                            const lWrist = getPoseLM(15);
                                            
                                            // Strict Filter: If Wrist is below Waist or Elbow confidence low -> Attention Pose
                                            // Simple Check: If Wrist Y > Elbow Y (Remember Y is inverted in 3D, but in MP: Y increases downwards)
                                            // Actually simpler: If lWrist is [0,0,0] OR Left Hand Missing -> Rest
                                            
                                            if (lShoulder[0] !== 0 && lElbow[0] !== 0 && lWrist[0] !== 0) {{
                                                solveBoneRotation(leftArm, lShoulder, lElbow, new THREE.Vector3(1, 0, 0));
                                                solveBoneRotation(leftForeArm, lElbow, lWrist, new THREE.Vector3(1, 0, 0));
                                                
                                                if (leftHand && frame.left_hand && frame.left_hand.length > 0) {{ 
                                                    solveHandOrientation(leftHand, getLeftHandLM(0), getLeftHandLM(5), getLeftHandLM(9), getLeftHandLM(17), 'left');
                                                }}
                                            }} else {{
                                                resetArmToRest(leftArm, leftForeArm, 'left');
                                            }}

                                            // 2. Right Arm Chain (VRM: -X)
                                            const rightArm = vrm.humanoid.getNormalizedBoneNode('rightUpperArm');
                                            const rightForeArm = vrm.humanoid.getNormalizedBoneNode('rightLowerArm');
                                            const rightHand = vrm.humanoid.getNormalizedBoneNode('rightHand');
                                            
                                            const rShoulder = getPoseLM(12);
                                            const rElbow = getPoseLM(14);
                                            const rWrist = getPoseLM(16);

                                            if (rShoulder[0] !== 0 && rElbow[0] !== 0 && rWrist[0] !== 0) {{
                                                solveBoneRotation(rightArm, rShoulder, rElbow, new THREE.Vector3(-1, 0, 0));
                                                solveBoneRotation(rightForeArm, rElbow, rWrist, new THREE.Vector3(-1, 0, 0));
                                                
                                                if (rightHand && frame.right_hand && frame.right_hand.length > 0) {{
                                                     solveHandOrientation(rightHand, getRightHandLM(0), getRightHandLM(5), getRightHandLM(9), getRightHandLM(17), 'right');
                                                }}
                                            }} else {{
                                                resetArmToRest(rightArm, rightForeArm, 'right');
                                            }}
                                            
                                            // 3. Head Rotation
                                            const head = vrm.humanoid.getNormalizedBoneNode('head');
                                            if (head) {{
                                                const leftEar = getPoseLM(7);
                                                const rightEar = getPoseLM(8);
                                                const yaw = (leftEar[0] - rightEar[0]) * 2.0; 
                                                head.rotation.y = THREE.MathUtils.lerp(head.rotation.y, yaw, 0.1);
                                            }}
                                        }}
                                    }}
                                    
                                    // Loop DNA
                                    document.getElementById('frame').textContent = (frameIdx+1) + '/' + DNA.length;
                                    frameIdx = (frameIdx + 1) % DNA.length;
                                }}
                            }}
                            renderer.render(scene, camera);
                        }}
                        animate();
                    </script>
                    </body></html>
                    """
                    st.components.v1.html(html_component, height=550)
                    st.caption(f"🎬 DNA Stream Active | **{' '.join(words)}**")

            col_orig, col_av, col_neo = st.columns(3)
            with col_orig:
                st.markdown("### 📽️ Source Benchmark")
                if os.path.exists(st.session_state['benchmark_path']):
                    st.video(st.session_state['benchmark_path'])
                    st.caption("Standard library stitching.")
                
            with col_av:
                st.markdown("### 🤖 Seamless Skeletal")
                if os.path.exists(st.session_state['skeletal_path']):
                    st.video(st.session_state['skeletal_path'])
                    st.caption("Internal 'Takhyeet' engine.")
                
            with col_neo:
                st.markdown("### 🦾 Neo-Avatar 3D")
                if os.path.exists(st.session_state['neo_path']):
                    st.video(st.session_state['neo_path'])
                    st.caption("Premium Volumetric Representation.")
                else:
                    st.warning("⚠️ Avatar render requires skeletal DNA.")

    # TAB 2: VIDEO TO TEXT
    with tab2:
        st.header("🎥 Sign Language Video to Text")
        uploaded_file = st.file_uploader("Upload video (.mp4)", type=["mp4", "avi", "mov"])
        
        if uploaded_file:
            temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())
            st.video(temp_path)
            
            if st.button("🔍 Recognize Sign"):
                with st.spinner("Analyzing landmarks..."):
                    label, confidence = core.predict_sign(temp_path)
                    if label:
                        st.success(f"🏆 Detected: {label} ({confidence:.1f}%)")
                    else:
                        st.error("❌ Detection failed.")

    st.markdown("---")
    st.markdown("Designed by **Ahmed Eltaweel** | AI Architect @ Konecta 🚀")

if __name__ == "__main__":
    main()
