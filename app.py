"""
Konecta SLT Platform: Digital Human Interface v2.0
Enterprise-grade bidirectional translation between Text/Speech and Pakistani Sign Language (PSL).
"""

import os
import time
import builtins
import tempfile
import streamlit as st
import shutil
import cv2
import sys
import threading
import io
import json

# ==============================================================================
# --- SYSTEM & INFRASTRUCTURE INITIALIZATION ---
# ==============================================================================
# 1. Path Redirection Layer (For Streamlit Cloud Write Support)
WRITABLE_BASE = os.path.join(tempfile.gettempdir(), "slt_persistent_storage")
APP_ROOT = os.path.abspath(os.getcwd()).replace("\\", "/")

# 2. Google Cloud Platform (GCP) Credentials
# Use a more robust absolute path detection
GCP_KEY_FILENAME = "service-account-key.json"
GCP_KEY_PATH = os.path.abspath(os.path.join(os.getcwd(), GCP_KEY_FILENAME)).replace("\\", "/")
GCP_ENABLED = False

if os.path.exists(GCP_KEY_PATH):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCP_KEY_PATH
    GCP_ENABLED = True
elif os.path.exists(os.path.join(tempfile.gettempdir(), GCP_KEY_FILENAME)):
    # Check in redirected storage too
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(tempfile.gettempdir(), GCP_KEY_FILENAME)
    GCP_ENABLED = True

# Recovery: Capture absolute original functions once and preserve them in sys
if not hasattr(sys, "_slt_orig"):
    sys._slt_orig = {
        "makedirs": os.makedirs,
        "mkdir": os.mkdir,
        "open": io.open, 
        "rename": os.rename,
        "replace": os.replace,
        "exists": os.path.exists,
        "isfile": os.path.isfile,
        "listdir": os.listdir
    }

_orig_makedirs = sys._slt_orig["makedirs"]
_orig_mkdir    = sys._slt_orig["mkdir"]
_orig_open     = sys._slt_orig["open"]
_orig_rename   = sys._slt_orig["rename"]
_orig_replace  = sys._slt_orig["replace"]
_orig_exists   = sys._slt_orig["exists"]
_orig_isfile   = sys._slt_orig["isfile"]
_orig_listdir  = sys._slt_orig["listdir"]

# Thread-local recursion guard
_slt_tls = threading.local()

def _ensure_parent(path):
    """Ensures parent directory exists for writable storage."""
    try:
        parent = os.path.dirname(path)
        if parent and not _orig_exists(parent):
            _orig_makedirs(parent, exist_ok=True)
    except:
        pass

def _redirect_path(path, is_write=False):
    if not path: return path
    
    # Normalize to absolute path for consistent comparison
    try:
        abs_p = os.path.abspath(path).replace("\\", "/")
    except:
        abs_p = str(path).replace("\\", "/")

    # Rule 1: Redirect any path inside site-packages/sign_language_translator
    if "sign_language_translator" in abs_p and "site-packages" in abs_p:
        rel = abs_p.split("sign_language_translator/")[-1]
        shadow = os.path.join(WRITABLE_BASE, rel).replace("\\", "/")
        if is_write:
            _ensure_parent(shadow)
            return shadow
        if _orig_exists(shadow):
            return shadow

    # Rule 2: Redirect any write attempt to the project folder (ReadOnly on SL Cloud)
    if abs_p.startswith(APP_ROOT) and not abs_p.startswith(WRITABLE_BASE.replace("\\", "/")):
        rel = os.path.relpath(abs_p, APP_ROOT).replace("\\", "/")
        # Avoid infinite nesting if rel is '.'
        if rel == ".": return abs_p
        shadow = os.path.join(WRITABLE_BASE, rel).replace("\\", "/")
        if is_write:
            _ensure_parent(shadow)
            return shadow
        if _orig_exists(shadow):
            return shadow

    # Rule 3: For existing redirects in /tmp, ensure parent exists on write
    if abs_p.startswith(WRITABLE_BASE.replace("\\", "/")) and is_write:
        _ensure_parent(abs_p)

    return path

def _patched_open(file, *args, **kwargs):
    if getattr(_slt_tls, 'active', False): 
        return _orig_open(file, *args, **kwargs)
    _slt_tls.active = True
    try:
        mode = args[0] if args else kwargs.get('mode', 'r')
        is_write = any(m in mode for m in ('w', 'a', '+', 'x'))
        return _orig_open(_redirect_path(file, is_write=is_write), *args, **kwargs)
    finally:
        _slt_tls.active = False

def _patched_exists(path):
    if getattr(_slt_tls, 'active', False): 
        return _orig_exists(path)
    _slt_tls.active = True
    try:
        target = _redirect_path(path)
        return _orig_exists(target) or _orig_exists(path)
    finally:
        _slt_tls.active = False

# Apply surgical patches
os.makedirs = lambda n, *a, **k: _orig_makedirs(_redirect_path(n, True), *a, **k) if not getattr(_slt_tls, 'active', False) else _orig_makedirs(n, *a, **k)
os.mkdir    = lambda p, *a, **k: _orig_mkdir(_redirect_path(p, True), *a, **k) if not getattr(_slt_tls, 'active', False) else _orig_mkdir(p, *a, **k)
builtins.open = _patched_open
os.rename   = lambda s, d, *a, **k: _orig_rename(_redirect_path(s), _redirect_path(d, True), *a, **k)
os.replace  = lambda s, d, *a, **k: _orig_replace(_redirect_path(s), _redirect_path(d, True), *a, **k)
os.path.exists = _patched_exists
os.path.isfile = lambda p: _orig_isfile(_redirect_path(p))
os.path.listdir  = lambda p: _orig_listdir(_redirect_path(p))

# Ensure WRITABLE_BASE exists
if not _orig_exists(WRITABLE_BASE):
    _orig_makedirs(WRITABLE_BASE, exist_ok=True)

# --- GOOGLE CLOUD SERVICE ACCOUNT INTEGRATION ---
if "gcp_service_account" in st.secrets:
    gcp_json_path = os.path.join(WRITABLE_BASE, "gcp_credentials.json")
    with _orig_open(gcp_json_path, "w") as f:
        json.dump(dict(st.secrets["gcp_service_account"]), f)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_json_path

# --- ENCODING FIX ---
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
    /* COMPUTER MODE FOR MOBILE (Desktop Feel) */
    @media (max-width: 1024px) {
        .main .block-container { min-width: 1000px !important; overflow-x: auto !important; padding-top: 2rem !important; }
        .stApp { min-width: 1000px !important; }
        header[data-testid="stHeader"] { min-width: 1000px !important; }
    }
</style>
""", unsafe_allow_html=True)

# Vocabulary Mapping (Stable 8 - Confirmed SLT Library Mappings)
PSL_VOCABULARY = {
    # These 8 words have verified labels in pk-dictionary-mapping.json
    "apple": "سیب",      # pk-hfad-1_apple
    "world": "دنیا",     # pk-hfad-1_world
    "good": "اچھا",      # pk-hfad-1_good
    "school": "اسکول",   # pk-hfad-1_school
    "mother": "ماں",     # pk-hfad-1_mother
    "father": "باپ",     # pk-hfad-1_papa
    "help": "مدد",       # pk-hfad-1_help
    "home": "گھر",       # pk-hfad-1_house
}

# Pending Vocabulary (Not in pk-dictionary-mapping.json - Awaiting Library Support)
PSL_PENDING = {
    "hello": "ہیلو", "salam": "سلام", "water": "پانی", 
    "food": "کھانا", "thanks": "شکریہ", "yes": "ہاں", "no": "نہیں"
}

# App Data Paths
DATA_DIR = os.path.join(WRITABLE_BASE, "app_internal_data")
os.makedirs(DATA_DIR, exist_ok=True)

@st.cache_resource
def get_slt_core_v2():
    """
    Singleton provider for the SignLanguageCore engine.
    Implements v2 architecture for skeletal DNA synthesis.
    """
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
        return True
    return False

def main():
    # --- GLOBAL SESSION STATE INITIALIZATION ---
    if "live_performance_mode" not in st.session_state:
        st.session_state.live_performance_mode = "⚡ High Performance (No Overlay)"
    if 'shared_sentence' not in st.session_state:
        st.session_state['shared_sentence'] = []
    if "recording" not in st.session_state:
        st.session_state.recording = False
    if "recorded_frames" not in st.session_state:
        st.session_state.recorded_frames = []
    
    # App UI Initialization
    
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
        core = get_slt_core_v2()
        renderer = get_avatar_renderer()
    
    if not load_or_train_core(core, translator):
        st.error("❌ Failed to initialize SLT Core.")
        st.stop()
    
    # --- CLEAN UI ---
    st.sidebar.success("💎 **Rigging Standard:** Elite Studio v2.0")
    
    if GCP_ENABLED:
        st.sidebar.markdown("✅ **GCP Production Mode:** Active")
        st.sidebar.caption("Infrastructure linked via Service Account.")
    else:
        st.sidebar.warning("⚠️ **GCP Mode:** Local/Sandbox")
    
    st.sidebar.info("🖥️ **Mode:** Forced Desktop (Computer View)")
    
    with st.sidebar.expander("🛠️ Final Infrastructure Doc"):
        st.markdown("[📄 Management Presentation](https://github.com/eltaweelac/streamlit_slt_app_v2_stable_v3_stable/blob/main/KONECTA_SLT_PRESENTATION.md)")
        st.caption("Includes WebRTC/TURN & Roadmap.")
    
    flip_opt = st.sidebar.checkbox("Flip Avatar (180°)", value=False)
    
    if st.sidebar.button("🧠 Force Retrain Engine"):
        with st.sidebar.status("🧠 Retraining with Expanded Vocab...", expanded=True) as status:
            st.write("📥 Syncing Landmarks...")
            core.build_landmark_dictionary(translator)
            st.write("🧪 Training RF Matrix...")
            if core.train_core():
                status.update(label="✅ Training Complete!", state="complete")
                st.success("Model updated with latest vocabulary!")
            else:
                status.update(label="❌ Training Failed", state="error")
    
    def preprocess_text(text, vocab):
        """Smart NLP Preprocessor: Fuzzy matching + Extended Lemmatization."""
        import string
        import difflib
        
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = text.lower().split()
        refined = []
        
        # Extended stop words for cleaner PSL output
        stop_words = {
            "am", "are", "is", "was", "were", "be", "been", "being",
            "a", "an", "the", "this", "that", "these", "those",
            "i", "me", "my", "you", "your", "he", "she", "it", "we", "they",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "very", "really", "just", "also", "and", "or", "but", "so",
            "will", "would", "could", "should", "can", "may", "might",
            "have", "has", "had", "do", "does", "did", "go", "going", "went"
        }
        
        vocab_keys = list(vocab.keys())
        
        for w in tokens:
            # Skip common stop words
            if w in stop_words and w not in vocab:
                continue
            
            # 1. Direct match
            if w in vocab:
                refined.append(w)
                continue
            
            # 2. Extended Lemmatization
            lemma = w
            # Verb forms
            if w.endswith("ing"): lemma = w[:-3]
            elif w.endswith("tion"): lemma = w[:-4]
            elif w.endswith("ed"): lemma = w[:-2]
            elif w.endswith("ied"): lemma = w[:-3] + "y"
            elif w.endswith("ies"): lemma = w[:-3] + "y"
            elif w.endswith("es"): lemma = w[:-2]
            elif w.endswith("er"): lemma = w[:-2]
            elif w.endswith("est"): lemma = w[:-3]
            elif w.endswith("ly"): lemma = w[:-2]
            elif w.endswith("s") and not w.endswith("ss"): lemma = w[:-1]
            
            if lemma in vocab:
                refined.append(lemma)
                continue
            
            # 3. Fuzzy Matching (80%+ similarity)
            matches = difflib.get_close_matches(w, vocab_keys, n=1, cutoff=0.8)
            if matches:
                refined.append(matches[0])
                continue
            
            # 4. Try fuzzy on lemma too
            matches = difflib.get_close_matches(lemma, vocab_keys, n=1, cutoff=0.8)
            if matches:
                refined.append(matches[0])
                continue
            
            # 5. Skip unknown words (don't add them to refined list)
            # This ensures only valid vocabulary words are processed
        
        return refined

    tab1, tab2 = st.tabs(["📝 Text → Video", "🎥 Video → Text"])
    
    # TAB 1: TEXT TO VIDEO
    with tab1:
        st.header("📝 Text to Sign Language Video")
        if 'text_input_val' not in st.session_state:
            st.session_state['text_input_val'] = ""

        text_input = st.text_input("Enter text (English):", value=st.session_state['text_input_val'], placeholder="e.g., hello world", key="main_input")
        
        # Enhanced Vocabulary Grid
        st.markdown("**📖 Supported Vocabulary (Active & Pending):**")
        all_keys = list(PSL_VOCABULARY.keys())
        available_words = core.get_available_words()
        
        grid_cols = st.columns(4)
        for idx, k in enumerate(all_keys):
            with grid_cols[idx % 4]:
                if k in available_words:
                    st.markdown(f"✅ `{k}`")
                else:
                    st.markdown(f"⏳ `{k}`")
        st.caption("💡 **Active (✅):** Ready to use. | **Pending (⏳):** Click 'Force Retrain' in sidebar to activate.")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            gen_btn = st.button("🚀 Generate Digital Human Output")
        with col2:
            st.warning("⚠️ **Voice Disclaimer:** Microphone access requires a local setup and may not be available on server-hosted environments.")
            if st.button("🎤 Use Voice Input"):
                with st.spinner("🎙️ Listening..."):
                    voice_text = core.speech_to_text()
                    if voice_text:
                        if voice_text.startswith("ERROR:"):
                            st.error(voice_text)
                        else:
                            st.info(f"🎤 Heard: **{voice_text}**")
                            text_input = voice_text
                            st.session_state['text_input_val'] = voice_text

        if gen_btn and text_input:
            with st.spinner("🧪 Transforming to Digital Avatar..."):
                # Use our new NLP preprocessor
                words = preprocess_text(text_input, PSL_VOCABULARY)
                
                # Audit Insight: Inform user if words were optimized
                original_tokens = text_input.lower().split()
                if len(words) < len(original_tokens) or any(w not in original_tokens for w in words):
                    st.toast("🔮 NLP: Sentence optimized for Sign Language.")
                
                st.session_state['last_words'] = words
                v_clips = []
                dna_list = []
                
                status_placeholder = st.empty() # Added for dynamic status updates
                for w in words:
                    if w in PSL_VOCABULARY:
                        # First check if DNA exists locally (most efficient)
                        dna = core.get_word_dna(w)
                        if dna is None:
                            status_placeholder.warning(f"⚠️ Word **{w}** not in landmarks. Skipping.")
                            continue
                        
                        try:
                            status_placeholder.info(f"🔍 Processing: **{w}**")
                            
                            # Use Urdu token directly (library's native format)
                            urdu_token = PSL_VOCABULARY[w]
                            try:
                                clip = translator.translate(urdu_token)
                            except:
                                clip = None
                            
                            if clip is None or len(clip) == 0:
                                # Fallback to English key
                                try:
                                    clip = translator.translate(w)
                                except:
                                    clip = None

                            if clip is not None and len(clip) > 0:
                                v_clips.append(clip)
                                dna_list.append(dna)
                                status_placeholder.success(f"✅ **{w}** ready.")
                            else:
                                status_placeholder.error(f"❌ Could not synthesize video for: **{w}**")
                        except Exception as e:
                            status_placeholder.error(f"❌ Error processing **{w}**: {e}")
                    else:
                        # Word was filtered out by NLP or not in vocab
                        pass
                
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
            
            # Character Selection HUD
            char_col1, char_col2 = st.columns([1, 1])
            with char_col1:
                cinema_mode = st.toggle("🎭 Activate Cinema Mode (3D VRM Client)", value=False)
            
            with char_col2:
                avatar_options = {
                    "🤖 Konecta Neo (Standard)": "5084648674725325209.vrm",
                    "🌟 Konecta Prime (High-Fidelity)": "4152045953412205614.vrm",
                    "🎬 Konecta Elite (Studio)": "VRM1_Constraint_Twist_Sample.vrm"
                }
                selected_avatar_name = st.selectbox("Select Digital Human:", list(avatar_options.keys()), index=0)
                vrm_path = avatar_options[selected_avatar_name]

            words = st.session_state.get('last_words', [])
            full_dna = st.session_state.get('full_dna')
            
            if cinema_mode:
                st.markdown(f"### 🎬 Cinema Mode: {selected_avatar_name}")
                # Use base keys directly for DNA retrieval to ensure multi-word stitching
                dna_json = core.get_words_dna_json(words)
                if dna_json:
                    import json
                    import base64
                    
                    # Embed local VRM model for stability
                    vrm_base64 = ""
                    if os.path.exists(vrm_path):
                        with open(vrm_path, "rb") as f:
                            vrm_base64 = base64.b64encode(f.read()).decode()
                    else:
                        st.error(f"❌ VRM Model '{vrm_path}' not found. Please ensure it is in the project directory.")
                        st.stop()
                    
                    st.info(f"🤖 **{selected_avatar_name} Bridge Active** | Transmitting Skeletal DNA...")
                    
                    html_component = """
                    <!DOCTYPE html>
                    <html><head>
                    <style>
                        * { margin:0; padding:0; box-sizing:border-box; }
                        body { background: linear-gradient(135deg, #05050a, #101018); display:flex; justify-content:center; align-items:center; min-height:100vh; font-family:'Inter',sans-serif; color:#fff; overflow:hidden; }
                        #canvas { width:100%; height:450px; border-radius:12px; box-shadow: 0 0 30px rgba(15, 157, 88, 0.1); }
                        .hud { display:flex; justify-content:space-around; margin-top:10px; padding:10px; background:rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05); border-radius:8px; backdrop-filter: blur(5px); }
                        .hud-item { text-align:center; }
                        .hud-label { font-size:0.65rem; color:#888; text-transform:uppercase; letter-spacing: 1px; }
                        .hud-value { font-size:1.0rem; font-weight:bold; color:#0f9d58; text-shadow: 0 0 10px rgba(15,157,88,0.5); }
                        #neural-overlay { 
                            position: absolute; top:0; left:0; width:100%; height:100%; 
                            pointer-events:none; background: linear-gradient(90deg, transparent, rgba(15,157,88,0.05), transparent);
                            background-size: 200% 100%; animation: scan 3s infinite linear; display:none;
                        }
                        @keyframes scan { 0% { background-position: 200% 0; } 100% { background-position: -200% 0; } }
                        .production-tag { position:absolute; top:20px; right:20px; background:rgba(15,157,88,0.9); padding:4px 8px; border-radius:4px; font-size:0.6rem; font-weight:bold; letter-spacing:1px; }
                    </style>
                    </head><body>
                    <div style="position:relative; width:100%; max-width:700px; padding:10px; background:rgba(255,255,255,0.02); border-radius:16px; border:1px solid rgba(255,255,255,0.05);">
                        <div id="neural-overlay"></div>
                        <div class="production-tag">NEURAL SYNTHESIS ACTIVE</div>
                        <canvas id="canvas"></canvas>
                        <div class="hud">
                            <div class="hud-item"><div class="hud-label">Engine</div><div class="hud-value">Google-MediaPipe-MP</div></div>
                            <div class="hud-item"><div class="hud-label">Status</div><div class="hud-value" id="status">Syncing DNA...</div></div>
                            <div class="hud-item"><div class="hud-label">Frame</div><div class="hud-value" id="frame">0/VAR_DNA_LEN</div></div>
                            <div class="hud-item"><div class="hud-label">Words</div><div class="hud-value">VAR_WORDS</div></div>
                        </div>
                    </div>
                    <script type="importmap">
                    {
                        "imports": {
                            "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
                            "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
                        }
                    }
                    </script>
                    <script type="module">
                        import * as THREE from 'three';
                        import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
                        import { VRMLoaderPlugin, VRMUtils } from 'https://cdn.jsdelivr.net/npm/@pixiv/three-vrm@2.0.6/lib/three-vrm.module.js';

                        const DNA = VAR_DNA_DATA;
                        const VAR_FLIP_AVATAR = VAR_FLIP_VAL;
                        let vrm, frameIdx = 0;
                        const canvas = document.getElementById('canvas');
                        const scene = new THREE.Scene();
                        scene.background = new THREE.Color(0x0a0a12); 
                        const camera = new THREE.PerspectiveCamera(35, canvas.clientWidth / canvas.clientHeight, 0.1, 100);
                        camera.position.set(0, 1.2, 2.5); // Unified stable view
                        camera.lookAt(0, 1.2, 0);

                        const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
                        renderer.setSize(canvas.clientWidth, canvas.clientHeight);
                        renderer.setPixelRatio(window.devicePixelRatio);
                        renderer.outputColorSpace = THREE.SRGBColorSpace; 
                        renderer.toneMapping = THREE.ACESFilmicToneMapping;
                        renderer.toneMappingExposure = 1.1;
                        
                        // --- ELITE STUDIO LIGHTING (STABLE) ---
                        const hemiLight = new THREE.HemisphereLight(0xffffff, 0x444444, 1.0);
                        scene.add(hemiLight);

                        const mainLight = new THREE.DirectionalLight(0xffffff, 1.5);
                        mainLight.position.set(5, 5, 5);
                        scene.add(mainLight);

                        const fillLight = new THREE.PointLight(0x0f9d58, 0.8, 10);
                        fillLight.position.set(-5, 2, 2);
                        scene.add(fillLight);

                        document.getElementById('neural-overlay').style.display = 'block';

                        const loader = new GLTFLoader();
                        loader.register(p => new VRMLoaderPlugin(p));
                        
                        const vrmData = "data:application/octet-stream;base64," + "VAR_VRM_BASE64";
                        
                        if ("VAR_VRM_BASE64") {
                            loader.load(vrmData, gltf => {
                                vrm = gltf.userData.vrm;
                                VRMUtils.rotateVRM0(vrm);
                                vrm.scene.traverse(obj => {
                                    if (obj.isMesh) {
                                        obj.material.colorSpace = THREE.SRGBColorSpace;
                                        // Restore original premium shaders (MToon compatibility)
                                        if (obj.material.transmission !== undefined) obj.material.transmission = 0;
                                    }
                                });
                                // Manual Flip Control
                                if (VAR_FLIP_AVATAR) {
                                    vrm.scene.rotation.y = Math.PI;
                                }
                                scene.add(vrm.scene);
                                document.getElementById('status').textContent = 'Konecta Rep Online';
                            }, null, e => { 
                                document.getElementById('status').textContent = 'Load Error: Corrupted Data';
                                console.error(e);
                            });
                        } else {
                             document.getElementById('status').textContent = 'Load Error: No Model Data';
                        }

                        const clock = new THREE.Clock();
                        // STABLE ELITE STABILIZER
                        const stabilizer = 0.08; 
                        
                        function solveBoneRotation(node, p_start, p_end, baseDir) {
                            if (!node || !p_start || !p_end) return;
                            const vTarget = new THREE.Vector3(
                                p_end[0] - p_start[0],
                                -(p_end[1] - p_start[1]), 
                                -(p_end[2] - p_start[2])
                            ).normalize();
                            const q = new THREE.Quaternion().setFromUnitVectors(baseDir, vTarget);
                            node.quaternion.slerp(q, stabilizer); 
                        }
                        
                        function solveHandOrientation(node, p_wrist, p_index, p_middle, p_pinky, side) {
                           if (!node || !p_wrist || !p_middle || !p_pinky || !p_index) return;
                           const toVec3 = (a) => new THREE.Vector3(a[0], -a[1], -a[2]);
                           const vWrist = toVec3(p_wrist);
                           const vIndex = toVec3(p_index);
                           const vMiddle = toVec3(p_middle);
                           const vPinky = toVec3(p_pinky);
                           const vForward = new THREE.Vector3().subVectors(vMiddle, vWrist).normalize();
                           const vSide = new THREE.Vector3().subVectors(vIndex, vPinky).normalize();
                           const vNormal = new THREE.Vector3().crossVectors(vForward, vSide).normalize();
                           const matrix = new THREE.Matrix4();
                           if (side === 'left') {
                               matrix.makeBasis(vForward, vNormal, vSide);
                           } else {
                               const vForwardR = vForward.clone().negate();
                               matrix.makeBasis(vForwardR, vNormal, vSide);
                           }
                           const qFinal = new THREE.Quaternion().setFromRotationMatrix(matrix);
                           node.quaternion.slerp(qFinal, stabilizer); 
                        }
                        
                        function resetArmToRest(arm, forearm, side) {
                            if (!arm || !forearm) return;
                            const qArmDown = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, 0, side === 'left' ? 1.5 : -1.5)); 
                            const speed = 0.1;
                            arm.quaternion.slerp(qArmDown, speed);
                            forearm.quaternion.slerp(new THREE.Quaternion(), speed); 
                        }

                        const fps = 30; // Restore 30fps fluidity
                        const timePerFrame = 1.0 / fps;
                        let timeAccumulator = 0.0;

                        function animate() {
                            requestAnimationFrame(animate);
                            const delta = clock.getDelta();
                            timeAccumulator += delta;
                            if (vrm) {
                                vrm.update(delta);
                                const time = clock.getElapsedTime();
                                const chest = vrm.humanoid.getNormalizedBoneNode('chest');
                                if (chest) {
                                    chest.rotation.z = Math.sin(time * 1.5) * 0.005; 
                                    chest.rotation.x = (Math.sin(time * 1.2) * 0.01) + 0.05;
                                }
                                if (DNA.length > 0 && timeAccumulator >= timePerFrame) {
                                    timeAccumulator %= timePerFrame;
                                    const frame = DNA[frameIdx];
                                    if (frame) {
                                        if (frame.expressions && vrm.expressionManager) {
                                            const expr = frame.expressions;
                                            vrm.expressionManager.setValue('happy', expr.happy || 0);
                                            vrm.expressionManager.setValue('surprised', expr.surprised || 0);
                                            vrm.expressionManager.setValue('angry', expr.angry || 0);
                                            vrm.expressionManager.setValue('blink', expr.blink || 0);
                                        }
                                        if (frame.pose) {
                                            const pose = frame.pose;
                                            const getPoseLM = (idx) => (pose.length > idx*3) ? [pose[idx*3], pose[idx*3+1], pose[idx*3+2]] : [0,0,0];
                                            const getLeftHandLM = (idx) => (frame.left_hand && frame.left_hand.length > idx*3) ? [frame.left_hand[idx*3], frame.left_hand[idx*3+1], frame.left_hand[idx*3+2]] : null;
                                            const getRightHandLM = (idx) => (frame.right_hand && frame.right_hand.length > idx*3) ? [frame.right_hand[idx*3], frame.right_hand[idx*3+1], frame.right_hand[idx*3+2]] : null;
                                            const leftArm = vrm.humanoid.getNormalizedBoneNode('leftUpperArm');
                                            const leftForeArm = vrm.humanoid.getNormalizedBoneNode('leftLowerArm');
                                            const leftHand = vrm.humanoid.getNormalizedBoneNode('leftHand');
                                            if (getPoseLM(11)[0] !== 0) {
                                                // SMOOTHING: Slower interpolation for arms (0.05)
                                                solveBoneRotation(leftArm, getPoseLM(11), getPoseLM(13), new THREE.Vector3(1, 0, 0));
                                                solveBoneRotation(leftForeArm, getPoseLM(13), getPoseLM(15), new THREE.Vector3(1, 0, 0));
                                                if (leftHand && getLeftHandLM(0)) {
                                                    solveHandOrientation(leftHand, getLeftHandLM(0), getLeftHandLM(5), getLeftHandLM(9), getLeftHandLM(17), 'left');
                                                }
                                            } else { resetArmToRest(leftArm, leftForeArm, 'left'); }
                                            const rightArm = vrm.humanoid.getNormalizedBoneNode('rightUpperArm');
                                            const rightForeArm = vrm.humanoid.getNormalizedBoneNode('rightLowerArm');
                                            const rightHand = vrm.humanoid.getNormalizedBoneNode('rightHand');
                                            if (getPoseLM(12)[0] !== 0) {
                                                solveBoneRotation(rightArm, getPoseLM(12), getPoseLM(14), new THREE.Vector3(-1, 0, 0));
                                                solveBoneRotation(rightForeArm, getPoseLM(14), getPoseLM(16), new THREE.Vector3(-1, 0, 0));
                                                if (rightHand && getRightHandLM(0)) {
                                                    solveHandOrientation(rightHand, getRightHandLM(0), getRightHandLM(5), getRightHandLM(9), getRightHandLM(17), 'right');
                                                }
                                            } else { resetArmToRest(rightArm, rightForeArm, 'right'); }
                                        }
                                    }
                                    document.getElementById('frame').textContent = (frameIdx+1) + '/' + DNA.length;
                                    frameIdx = (frameIdx + 1) % DNA.length;
                                }
                            }
                            renderer.render(scene, camera);
                        }
                        animate();
                    </script>
                    </body></html>
                    """.replace("VAR_WORDS", ' '.join(words)) \
                       .replace("VAR_DNA_LEN", str(len(dna_json))) \
                       .replace("VAR_DNA_DATA", json.dumps(dna_json)) \
                       .replace("VAR_VRM_BASE64", vrm_base64) \
                       .replace("VAR_FLIP_VAL", "true" if flip_opt else "false")
                    
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
                    st.caption("Concatenative Synthesis Engine.")
                
            with col_neo:
                st.markdown("### 🦾 Neo-Avatar 3D")
                if os.path.exists(st.session_state['neo_path']):
                    st.video(st.session_state['neo_path'])
                    st.caption("Premium Volumetric Representation.")
                else:
                    st.warning("⚠️ Avatar render requires skeletal DNA.")

    # --- TAB 2: VIDEO TO TEXT ---
    with tab2:
        st.header("🎥 Video to Sign Language Text")
        
        # --- SHARED SENTENCE BUILDER UI ---
            
        if st.session_state['shared_sentence']:
            st.info(f"📝 **Sentence Builder:** {' '.join(st.session_state['shared_sentence'])}")
            c1, c2, c3 = st.columns(3)
            if c1.button("🗑️ Clear Sentence", key="btn_clr_shared"):
                st.session_state['shared_sentence'] = []
                st.rerun()
            if c2.button("🚀 Push to Avatar", key="btn_push_shared"):
                st.session_state['text_input_val'] = " ".join(st.session_state['shared_sentence'])
                st.success("✅ Sequence synced to Tab 1!")
            if c3.button("🔊 Speak Sentence", key="btn_speak_shared"):
                sentence = " ".join(st.session_state['shared_sentence'])
                try:
                    from gtts import gTTS
                    import base64
                    tts = gTTS(text=sentence, lang='en')
                    audio_file = os.path.join(tempfile.gettempdir(), "tts_output.mp3")
                    tts.save(audio_file)
                    with open(audio_file, "rb") as f:
                        audio_bytes = f.read()
                    st.audio(audio_bytes, format="audio/mp3")
                    st.success(f"🔊 Speaking: **{sentence}**")
                except Exception as e:
                    st.error(f"❌ TTS Error: {e}")
            st.markdown("---")

        # Camera mode selection
        st.subheader("🎥 Live Sign Analysis")
        camera_mode = st.radio(
            "Choose camera method:",
            ["📷 Quick Capture (Recommended)", "🌐 Live Stream (Advanced)"],
            horizontal=True,
            help="Quick Capture works everywhere. Live Stream requires WebRTC support."
        )
        
        if camera_mode == "📷 Quick Capture (Recommended)":
            st.markdown("**Instructions:** Click the camera button below, make your sign, and capture the frame.")
            st.warning("⚠️ Note: Single-frame capture works best for static signs. For dynamic signs, use **Upload Video** below.")
            
            img_file = st.camera_input("📷 Capture your sign gesture")
            
            if img_file is not None:
                # Decode image
                import numpy as np
                file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                if img is not None:
                    with st.spinner("🧠 Analyzing captured frame..."):
                        try:
                            # Extract landmarks from single frame using MediaPipe
                            mp_holistic = mp.solutions.holistic
                            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                results = holistic.process(img_rgb)
                                
                                # Build landmark vector same as video extraction
                                landmarks = []
                                
                                # Left hand (21 points)
                                if results.left_hand_landmarks:
                                    for lm in results.left_hand_landmarks.landmark:
                                        landmarks.extend([lm.x, lm.y, lm.z])
                                else:
                                    landmarks.extend([0.0] * 63)
                                
                                # Right hand (21 points)
                                if results.right_hand_landmarks:
                                    for lm in results.right_hand_landmarks.landmark:
                                        landmarks.extend([lm.x, lm.y, lm.z])
                                else:
                                    landmarks.extend([0.0] * 63)
                                
                                # Pose (33 points - but we use 25 key points)
                                if results.pose_landmarks:
                                    key_indices = [0,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
                                    for idx in key_indices:
                                        lm = results.pose_landmarks.landmark[idx]
                                        landmarks.extend([lm.x, lm.y, lm.z])
                                else:
                                    landmarks.extend([0.0] * 45)
                                
                                # Now predict using landmarks
                                if len(landmarks) > 0 and sum(landmarks) != 0:
                                    landmarks_array = np.array(landmarks).reshape(1, -1)
                                    label, confidence = core.predict_from_landmarks(landmarks_array[0])
                                    
                                    if label and confidence > 0.5:
                                        st.success(f"🏆 Recognized: **{label}** (Confidence: {confidence:.1%})")
                                        
                                        # Add to shared sentence
                                        if not st.session_state['shared_sentence'] or st.session_state['shared_sentence'][-1] != label:
                                            st.session_state['shared_sentence'].append(label)
                                    else:
                                        st.warning("⚠️ Could not recognize the sign. Try again with a clearer gesture.")
                                else:
                                    st.warning("⚠️ No hands detected in frame. Please show your hands clearly.")
                        except Exception as e:
                            st.error(f"❌ Analysis Error: {e}")
                else:
                    st.error("❌ Could not process the image. Please try again.")
            
            st.info("💡 **Tip:** For video analysis, use the file uploader below or run `standalone_live.py` for desktop mode.")
        
        else:  # Live Stream mode
            live_mode = st.toggle("⚡ Enable Live Streaming", value=False)
            
            if live_mode:
                st.subheader("🌐 Live Hybrid Recording")
                st.markdown("1. **Start Stream** | 2. **Record Sign** | 3. **Get Transcription**")
                
                try:
                    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
                    import av
                    import queue

                    st.sidebar.markdown("---")
                    st.session_state.live_performance_mode = st.sidebar.radio(
                        "🎭 Live Analysis Mode",
                        ["🧪 AI Intelligent (Live Processing)", "⚡ High Performance (No Overlay)"],
                        help="Intelligent mode runs landmark extraction in real-time but may lag on slow CPUs."
                    )
                    
                    class SignProcessor:
                        def __init__(self, mode):
                            self.frame_queue = queue.Queue()
                            self.mode = mode
                            self.frame_count = 0
                            try:
                                import mediapipe as mp
                                self.hands = mp.solutions.hands.Hands(
                                    static_image_mode=False,
                                    max_num_hands=2,
                                    min_detection_confidence=0.5,
                                    min_tracking_confidence=0.5
                                )
                                self.mp_drawing = mp.solutions.drawing_utils
                            except Exception as e:
                                self.hands = None
                                print(f"MediaPipe Init Error: {e}")
                            
                            self.last_prediction = ""
                            self.last_confidence = 0.0

                        def recv(self, frame):
                            try:
                                img = frame.to_ndarray(format="bgr24")
                                self.frame_count += 1
                                
                                # PERFORMANCE OPTIMIZATION: Use High-Speed Hands track for overlay
                                if self.mode == "🧪 AI Intelligent (Live Processing)" and self.hands and self.frame_count % 2 == 0:
                                    try:
                                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                        results = self.hands.process(img_rgb)
                                        
                                        if hasattr(results, 'multi_hand_landmarks') and results.multi_hand_landmarks:
                                            for hand_landmarks in results.multi_hand_landmarks:
                                                self.mp_drawing.draw_landmarks(
                                                    img, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                                        
                                        cv2.putText(img, "STREAMING ACTIVE", (30, 30), 
                                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                    except: pass

                                # Put in queue for recording/transcribing
                                if not self.frame_queue.full():
                                    self.frame_queue.put(img)
                                
                                return av.VideoFrame.from_ndarray(img, format="bgr24")
                            except Exception as e:
                                # SILENT FAIL: Don't show red error on transient frame drops
                                return frame

                    # Infrastructure Selection: Diversified STUN for Corporate Compatibility (KSA & Global)
                    ice_servers = [
                        {"urls": ["stun:stun.l.google.com:19302"]},
                        {"urls": ["stun:stun1.l.google.com:19302"]},
                        {"urls": ["stun:stun.services.mozilla.com"]},
                        {"urls": ["stun:stun.l.google.com:19305"]},
                    ]
                    
                    # If user provides TURN credentials in secrets, prioritize them
                    if "GCP_TURN_SERVER" in os.environ:
                        ice_servers.append({
                            "urls": [os.environ["GCP_TURN_SERVER"]],
                            "username": os.environ.get("GCP_TURN_USER", ""),
                            "credential": os.environ.get("GCP_TURN_PASS", "")
                        })

                    webrtc_ctx = webrtc_streamer(
                        key="slt-live-radical",
                        mode=WebRtcMode.SENDRECV,
                        rtc_configuration=RTCConfiguration({"iceServers": ice_servers}),
                        video_processor_factory=lambda: SignProcessor(st.session_state.live_performance_mode),
                        media_stream_constraints={"video": True, "audio": False},
                        async_processing=True,
                    )

                    with st.expander("🛠️ Live Stream Debug Info"):
                        st.write(f"Connection State: `{webrtc_ctx.state.playing}`")
                        st.write(f"GCP Mode: `{'Production' if GCP_ENABLED else 'Local'}`")
                        if not GCP_ENABLED:
                            st.info("💡 If video stays black, your corporate firewall is likely blocking STUN traffic. Please follow the GCP Infrastructure Setup guide to deploy a TURN server.")

                    col1, col2 = st.columns(2)
                    
                    if webrtc_ctx.video_processor:
                        if not st.session_state.recording:
                            if col1.button("🔴 Start Recording", use_container_width=True):
                                st.session_state.recording = True
                                st.session_state.recorded_frames = []
                                st.rerun()
                        else:
                            if col1.button("⏹️ Stop & Transcribe", use_container_width=True):
                                st.session_state.recording = False
                                # Drain queue into session state with progress
                                with st.status("📥 Fetching frames...", expanded=False) as status:
                                    while not webrtc_ctx.video_processor.frame_queue.empty():
                                        st.session_state.recorded_frames.append(webrtc_ctx.video_processor.frame_queue.get())
                                    status.update(label=f"✅ {len(st.session_state.recorded_frames)} frames captured", state="complete")
                                
                                if len(st.session_state.recorded_frames) > 10:
                                    with st.spinner("🧠 Analyzing Sign Sequence (This may take a moment)..."):
                                        # Save to temp file - BYPASS THE PATCHED WRITER to skip slow ffmpeg
                                        temp_vid = os.path.join(tempfile.gettempdir(), f"live_rec_{int(time.time())}.mp4")
                                        h, w, _ = st.session_state.recorded_frames[0].shape
                                        # Use _orig_VideoWriter directly to skip the H.264 optimization (not needed for analysis)
                                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                                        out = _orig_VideoWriter(temp_vid, fourcc, 20.0, (w, h))
                                        for f in st.session_state.recorded_frames:
                                            out.write(f)
                                        out.release()
                                        
                                        # Predict using the temporal engine
                                        labels, conf = core.predict_sentence(temp_vid)
                                        if labels:
                                            st.session_state.live_label = " ".join(labels)
                                            # Add to shared sentence
                                            for lbl in labels:
                                                if not st.session_state['shared_sentence'] or st.session_state['shared_sentence'][-1] != lbl:
                                                    st.session_state['shared_sentence'].append(lbl)
                                        else:
                                            st.error("❌ Recognition failed. Try signing closer to the camera with distinct pauses.")
                                else:
                                    st.warning("⚠️ Recording too short. Please capture at least 1 second of signing.")
                                st.rerun()

                    if "live_label" in st.session_state and st.session_state.live_label:
                        st.success(f"🏆 Recognized: **{st.session_state.live_label}**")
                    
                    if st.session_state.recording:
                        st.toast("🎥 Recording in progress...")
                        # In a real app we'd drain the queue periodically to avoid OOM
                        # For POC we drain when stopping.
                        while webrtc_ctx.video_processor and not webrtc_ctx.video_processor.frame_queue.empty():
                            st.session_state.recorded_frames.append(webrtc_ctx.video_processor.frame_queue.get())

                except Exception as e:
                    st.error(f"❌ Live Stream Error: {e}")
                    st.info("💡 Try using the **Quick Capture** mode instead for better compatibility.")
            
            else:
                st.subheader("📁 Upload Video for Analysis")
                st.markdown("""**Instructions:** Drop a clear video clip here to recognize the sign.""")
                
                if 'last_results' not in st.session_state:
                    st.session_state['last_results'] = {}

                uploaded_file = st.file_uploader("Upload Sign Clip", type=["mp4", "avi", "mov"], key="vid_uploader")
                
                if uploaded_file:
                    file_id = f"{uploaded_file.name}_{uploaded_file.size}"
                    
                    # Use a specific temp path for persistence during optimization
                    temp_path = os.path.join(tempfile.gettempdir(), f"upload_{file_id}.mp4")
                    
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # CRITICAL FIX: Optimize mobile video for OpenCV compatibility
                    with st.spinner("🎬 Optimizing Video for Analysis..."):
                        _optimize_video_for_web(temp_path)
                    
                    st.video(temp_path)
                    
                    # Two buttons: one for text only, one for text + speech
                    col_btn1, col_btn2 = st.columns(2)
                    
                    if col_btn1.button("🔍 Recognize Text", key="btn_recognize"):
                        with st.spinner("🧠 Analyzing Sign Sequences..."):
                            labels, confidence = core.predict_sentence(temp_path)
                            if labels:
                                sentence = " ".join(labels)
                                result_text = f"🏆 Sequence: **{sentence}** ({confidence:.1f}%)"
                                st.session_state['last_results'][file_id] = result_text
                                
                                # Add new words to shared sentence if they aren't already there in order
                                for label in labels:
                                    if not st.session_state['shared_sentence'] or st.session_state['shared_sentence'][-1] != label:
                                        st.session_state['shared_sentence'].append(label)
                            else:
                                st.error("❌ Recognition failed. Please try a clearer video with distinct pauses between signs.")
                    
                    # NEW: One-click Video → Speech conversion
                    if col_btn2.button("🔊 Recognize & Speak", key="btn_recognize_speak"):
                        with st.spinner("🧠 Analyzing Sign Language Video..."):
                            labels, confidence = core.predict_sentence(temp_path)
                            if labels:
                                sentence = " ".join(labels)
                                result_text = f"🏆 Sequence: **{sentence}** ({confidence:.1f}%)"
                                st.session_state['last_results'][file_id] = result_text
                                
                                # Add to shared sentence
                                for label in labels:
                                    if not st.session_state['shared_sentence'] or st.session_state['shared_sentence'][-1] != label:
                                        st.session_state['shared_sentence'].append(label)
                                
                                # Automatically convert to speech
                                st.success(result_text)
                                with st.spinner("🔊 Converting to Speech..."):
                                    try:
                                        from gtts import gTTS
                                        tts = gTTS(text=sentence, lang='en')
                                        audio_file = os.path.join(tempfile.gettempdir(), f"tts_{file_id}.mp3")
                                        tts.save(audio_file)
                                        with open(audio_file, "rb") as f:
                                            audio_bytes = f.read()
                                        st.audio(audio_bytes, format="audio/mp3", autoplay=True)
                                        st.info(f"🎤 **Speaking:** {sentence}")
                                    except Exception as e:
                                        st.error(f"❌ TTS Error: {e}")
                            else:
                                st.error("❌ Recognition failed. Please try a clearer video with distinct pauses between signs.")
                    
                    # Persist result display even after button click
                    if file_id in st.session_state['last_results']:
                        st.success(st.session_state['last_results'][file_id])

    st.markdown("---")
    st.markdown("Designed by **Ahmed Eltaweel** | AI Architect @ Konecta 🚀")

if __name__ == "__main__":
    main()
