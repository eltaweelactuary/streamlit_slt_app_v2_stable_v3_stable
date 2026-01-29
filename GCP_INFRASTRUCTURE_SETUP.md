# ☁️ Google Cloud Run: Deployment Guide v3.0

The application is now fully upgraded to a **FastAPI + Docker** architecture, optimized for high-performance deployment on Google Cloud.

## ✅ Readiness Audit
- [x] **Dockerfile:** Standardized for FastAPI (No Streamlit overhead).
- [x] **.dockerignore:** Implemented to exclude local assets and sensitive files.
- [x] **main.py:** Configured to listen on dynamic `$PORT` for Cloud Run compliance.
- [x] **Frontend:** Client-side MediaPipe processing to reduce server load.

---

## 🚀 Deployment Steps (4 Commands)

Follow these steps in your terminal (ensure you have [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) installed):

### Step 1: Initialize Local Variables
Set your project ID for easy copying:
```powershell
$PROJECT_ID = "YOUR_GCP_PROJECT_ID"
```

### Step 2: Authenticate with Google Cloud
```powershell
gcloud auth login
gcloud config set project $PROJECT_ID
```

### Step 3: Build & Upload Image (Google Artifact Registry)
This command packages the app and uploads it to Google's secured registry:
```powershell
gcloud builds submit --tag gcr.io/$PROJECT_ID/konecta-slt-v3
```

### Step 4: Deploy to Cloud Run
This command starts the service and makes it public:
```powershell
gcloud run deploy konecta-slt --image gcr.io/$PROJECT_ID/konecta-slt-v3 --platform managed --region us-central1 --allow-unauthenticated
```

---

## 🛠️ Performance Optimization for GCP
The app uses **`/tmp/slt_persistent_storage`** for on-the-fly assets, which is compatible with Cloud Run's ephemeral filesystem. For production use with multiple instances, we recommend mounting a **Google Cloud Storage (GCS) bucket**.

Designed by **Antigravity AI** for **Ahmed Eltaweel | Konecta 🚀**
