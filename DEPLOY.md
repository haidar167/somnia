# 🌐 Deployment Guide: Deploying THE SPECIMEN 24/7

This guide walks you through deploying **THE SPECIMEN** to free cloud hosting so the organism lives permanently on the internet with a public URL and WebSocket telemetry stream.

---

## 🚀 Option 1: Render.com (Recommended — Full WebSocket Support)

Render provides free cloud hosting with native WebSocket support and automatic GitHub deployments.

### Step-by-Step Instructions:

1. **Sign in to Render**:
   - Go to [render.com](https://render.com/) and sign in with your GitHub account (`haidar167`).

2. **Create New Web Service**:
   - Click the **"New +"** button in the dashboard $\to$ Select **"Web Service"**.
   - Under *Connect a repository*, choose `haidar167/somnia`.

3. **Configure Service Settings**:
   - **Name**: `the-specimen` (or `somnia-specimen`)
   - **Region**: Choose the closest region (e.g. `Frankfurt`, `Oregon`, or `Ohio`)
   - **Branch**: `master`
   - **Runtime**: Select **`Docker`** (Render will automatically detect the `Dockerfile`)
   - **Instance Type**: Select **`Free`** (0.5 CPU, 512 MB RAM)

4. **Environment Variables** (Optional / Default):
   - `PYTHONUNBUFFERED` = `1`
   - Render automatically injects `PORT` (defaulting to `10000` or `8000`).

5. **Deploy**:
   - Click **"Create Web Service"**.
   - Render will build the Docker container and start the uvicorn server.
   - Once complete, your public URL will be ready: `https://the-specimen.onrender.com`.

---

## 🤗 Option 2: Hugging Face Spaces (Fallback)

If you prefer Hugging Face:

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) $\to$ Click **"Create new Space"**.
2. Space Name: `the-specimen`
3. License: `MIT`
4. Space SDK: Choose **`Docker`** $\to$ **Blank**.
5. Connect your GitHub repository `haidar167/somnia` or push via Git.
6. Hugging Face will build the container and provide a live URL at `https://huggingface.co/spaces/haidar167/the-specimen`.

---

## ⚡ Mitigating Free-Tier Cold Starts (Keep-Alive Ping)

### Free-Tier Cloud Behavior:
- Render free instances spin down after ~15 minutes of inactivity to conserve cloud resources.
- When a new visitor accesses the URL, the container performs a **cold start** taking ~30 to 50 seconds.

### Free Keep-Alive Setup:
To keep the organism awake during daylight hours:
1. Go to [cron-job.org](https://cron-job.org/) (free service) or [uptimerobot.com](https://uptimerobot.com/).
2. Create a new cron job:
   - **URL**: `https://<your-render-subdomain>.onrender.com/health`
   - **Execution Schedule**: Every 10 minutes.
   - **HTTP Method**: `GET`
3. *Honest Tradeoff*: Render free accounts provide 750 free instance hours per month (enough for 1 service running 24/7 continuously).

---

## 💾 State Persistence & Ephemeral Disk Notes

- **What Persists**:
  - The SQLite database (`data/specimen_state.db`) records visitor interactions, dream archives, event logs, and memory buffers during container runtime.
  - Across software restarts (e.g. server crash or worker reload), SQLite restores the full history.
- **Ephemeral Free-Tier Redepoys**:
  - On free cloud tiers, redeploying the Docker image creates a fresh filesystem unless a persistent disk volume is attached.
  - If the database is reset on a fresh container build, the organism auto-bootstraps from seed 0 in 5 seconds and resumes living immediately.

---

## 🧪 Verifying Live Deployment

Once deployed, verify your service:
```bash
# 1. Healthcheck verification
curl -s https://<your-subdomain>.onrender.com/health

# Response:
# {"status":"alive","specimen_id":"SPECIMEN #001","uptime_seconds":120,"total_feeds":5,"known_visitors":2}
```
Open `https://<your-subdomain>.onrender.com` in your browser (desktop or mobile) to interact with the live console!
