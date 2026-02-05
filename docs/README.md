# 🤟 Next-Gen Sign Language Platform (v2.0 SOTA)

A state-of-the-art, bidirectional sign language translation and production platform featuring **Digital Human Avatar Synthesis** and **GCP Infrastructure**.

## 🌟 Key Features

| Feature | Technology | Description |
|-----------|-------------|-------------|
| **Digital Human Avatar** | 3D VRM Rigging | Transforms benchmark videos into stylized, noise-free sign language avatars. |
| **Neural Synthesis** | Three.js Studio | Simulates high-fidelity production rendering with cinematic lighting and skeletal DNA mapping. |
| **Dual-Mode Camera** | WebRTC & QuickCapture | Flexible live recognition for both cloud (st.camera_input) and local environments. |
| **High-Perf Desktop** | OpenCV Standalone | `standalone_live.py`: Zero-latency desktop app for real-time production workflows. |
| **Omni-channel** | gTTS & Pyttsx3 | Bidirectional: Text ↔ Sign ↔ Speech. Includes a "Speak" button for synthesized voice. |
| **GCP Ready** | Service Account | Integrated Google Cloud SDK logic for enterprise TURN servers and Vertex AI scaling. |

## 🚀 Quick Start

### 🌐 Cloud (Streamlit)
1. Fork this repo.
2. Connect to [Streamlit Cloud](https://share.streamlit.io/).
3. Set `app.py` as the main entry point.

### 🖥️ Desktop (Real-time Mvp)
For the best performance (30+ FPS), run the standalone desktop version:
```bash
python standalone_live.py
```

## 🏗️ Architecture: The Unified DNA Core
The system utilizes a **Unified Landmark Representation (CLR)**. Every sign is stored as a "Skeletal DNA" matrix, allowing for seamless stitching and high-fidelity avatar rendering across different digital character models (Neo, Prime, Elite).

## 📊 Core Vocabulary (Stable 8)
**Active:** `apple, world, good, school, mother, father, help, home`

---
**Designed by:** Ahmed Eltaweel | AI Solutions Architect @ Konecta 🚀
**Infrastructure:** GCP Service Account Integrated (Production Ready)
