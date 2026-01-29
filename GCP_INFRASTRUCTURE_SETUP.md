# 🌐 GCP Infrastructure Setup Guide: Enterprise Live Streaming

To leverage the full power of **Google Cloud** and fix the "lag/freezing" issues in corporate networks (Konecta), you must deploy a dedicated **TURN Server**. 

Your `service-account-key.json` is the key to automating this deployment.

## 🚀 1. Deploying the TURN Server (GCP Compute Engine)

A TURN server acts as a relay for video data, bypassing firewalls that block standard WebRTC.

### Step-by-Step Commands:

1. **Enable Compute Engine API:**
   ```bash
   gcloud services enable compute.googleapis.com
   ```

2. **Create a specialized VM instance (e2-micro is enough):**
   ```bash
   gcloud compute instances create turn-server-konecta \
       --zone=us-central1-a \
       --machine-type=e2-micro \
       --tags=turn-server \
       --metadata=startup-script="#! /bin/bash
   apt-get update
   apt-get install -y coturn
   echo 'TURNSERVER_ENABLED=1' > /etc/default/coturn
   cat <<EOF > /etc/turnserver.conf
   listening-port=3478
   fingerprint
   lt-cred-mech
   user=konecta_user:k0necta_p@ss
   realm=konecta.ai
   EOF
   systemctl restart coturn"
   ```

3. **Open Firewall Ports:**
   ```bash
   gcloud compute firewall-rules create allow-turn \
       --allow=udp:3478,tcp:3478 \
       --target-tags=turn-server
   ```

## 🧠 2. Linking to the App

Once deployed, get the External IP of your instance:
```bash
gcloud compute instances list
```

Add these to your Streamlit Secrets or Environment Variables:
- `GCP_TURN_SERVER`: `turn:<YOUR_GCP_IP>:3478`
- `GCP_TURN_USER`: `konecta_user`
- `GCP_TURN_PASS`: `k0necta_p@ss`

## 💎 Why this uses "Google Power"?
- **Google Global Network:** Video data travels through Google's premium fiber-optic network instead of the public internet.
- **Dedicated Resources:** The processing is backed by GCP's high-uptime infrastructure, ensuring the stream never "hangs" due to network relay issues.
- **Security:** Authenticated via the Service Account you provided.
