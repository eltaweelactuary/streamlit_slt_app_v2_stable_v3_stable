# 🛠️ GCP Deployment Troubleshooting

The `PERMISSION_DENIED` error during deployment is a common security restriction in corporate environments. Below is the breakdown of what to request from your Admin and how to bypass the issue immediately.

## 🔑 Checklist for the Admin
If you want the **"Automatic Deployment from GitHub"** to work, the Admin needs to grant you (and the service account) these roles:

1. **For Your Account:**
   - `roles/cloudbuild.builds.editor` (Cloud Build Editor)
   - `roles/run.admin` (Cloud Run Admin)
   - `roles/iam.serviceAccountUser` (Service Account User)

2. **For the Service Account** (`...-compute@developer.gserviceaccount.com`):
   - `roles/cloudbuild.builds.builder`

---

## 🚀 The "Fast-Track" Manual Deployment
If you cannot wait for the Admin, follow these steps in **Cloud Shell** to deploy the code directly. This bypasses the automatic trigger system.

### 1. Prepare the environment
```bash
# Clone the latest code
git clone https://github.com/eltaweelactuary/streamlit_slt_app_v2_stable_v3_stable.git
cd streamlit_slt_app_v2_stable_v3_stable

# Configure Docker permissions
gcloud auth configure-docker
```

### 2. Build and Push the Image
```bash
# Build the container (v3 represents our restructured version)
docker build -t gcr.io/eg-konecta-sandbox/konecta-slt:v3 .

# Push to Google's Registry
docker push gcr.io/eg-konecta-sandbox/konecta-slt:v3
```

### 3. Deploy to Cloud Run
```bash
gcloud run deploy stlv140 \
  --image gcr.io/eg-konecta-sandbox/konecta-slt:v3 \
  --region europe-west1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1
```

---
Designed by **Antigravity AI** for **Ahmed Eltaweel | Konecta 🚀**
