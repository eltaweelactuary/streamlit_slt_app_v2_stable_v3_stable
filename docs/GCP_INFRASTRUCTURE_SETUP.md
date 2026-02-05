# ☁️ Google Cloud Run: Simplified Web Deployment Guide

This guide is designed for direct deployment using the **Google Cloud Console UI**. No local terminal or SDK installation is required.

## 🚀 Step-by-Step Deployment (Console UI)

1. **Access the Project:**
   Go to your [Google Cloud Console](https://console.cloud.google.com/run?project=eg-konecta-sandbox).

2. **Create Service:**
   - Click the **"+ CREATE SERVICE"** button at the top.

3. **Configure Service:**
   - **Deployment Method:** Select **"Continuously deploy new revisions from a source repository"**.
   - Click **"SET UP WITH CLOUD BUILD"**.
   - **Repository:** Link your GitHub repository (`ahmedeltaweel-wq/streamlit_slt_app_v2_stable_v3_stable`).
   - **Build Configuration:** Select **"Dockerfile"**. The system will automatically find the `Dockerfile` in your root folder.

4. **Service Settings:**
   - **Service Name:** `konecta-slt`
   - **Region:** `us-central1` (or your preferred region).
   - **Authentication:** Select **"Allow unauthenticated invocations"** (to make the link public).

5. **Advanced Settings (Optional but Recommended):**
   - Click the **"Container, Networking, Security"** expander.
   - **Container Port:** Set to `8080`.
   - **Resources:** 2 GiB Memory / 1 vCPU is recommended for MediaPipe processing.

6. **Deploy:**
   - Click **"CREATE"**.

---

## 🏁 Result
Google Cloud will now build your container and provide a **Public URL** (e.g., `https://konecta-slt-xxxx-uc.a.run.app`). 

**That's it! Your Enterprise Sign Language Translator is now live.**

---

## 🔍 Pre-Flight Check (Files Audited)
- ✅ **`Dockerfile`**: Confirmed optimized for Cloud Build (runs `backend/server.py`).
- ✅ **`backend/server.py`**: Confirmed environment-aware (Port 8080).
- ✅ **`static/`**: All frontend assets (HTML/CSS/JS) are ready.
- ✅ **`assets/`**: VRM models and classifiers correctly placed.

---

**✅ The Solution (Use Your JSON Key):**
Since you have a Service Account JSON file, this is the **best** method.

1. **Upload the JSON file:**
   - In Cloud Shell, click the "Three Dots" menu (⋮) -> **Upload**.
   - Select your JSON file from your computer.

### �️ ULTIMATE FIX: Manual Docker Build (Bypass Cloud Build)

Since the system cannot "build" the code automatically due to missing permissions, we will **build it ourselves** inside Cloud Shell and simply upload the finished result.

**Copy and paste these 5 commands (This bypasses the error):**

```bash
# 1. Authorize Docker to push to Google Cloud
gcloud auth configure-docker

# 2. Build the Container Image locally in Cloud Shell
# (This happens on your machine, not Google's build servers)
docker build -t gcr.io/eg-konecta-sandbox/konecta-slt .

# 3. Push the Image to the Registry
docker push gcr.io/eg-konecta-sandbox/konecta-slt

# 4. Deploy the Pre-Built Image
gcloud run deploy konecta-slt --image gcr.io/eg-konecta-sandbox/konecta-slt --platform managed --region us-central1 --allow-unauthenticated
```

Designed by **Antigravity AI** for **Ahmed Eltaweel | Konecta 🚀**
