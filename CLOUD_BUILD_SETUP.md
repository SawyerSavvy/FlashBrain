# Cloud Build Setup Guide

## Setup Cloud Build with GitHub (Recommended)

This approach automatically deploys your agents whenever you push to GitHub.

### Step 1: Connect GitHub to Cloud Build

1. **Go to Cloud Build** in Google Cloud Console
   - https://console.cloud.google.com/cloud-build/triggers

2. **Click "Connect Repository"**
   - Select source: **GitHub**
   - Authenticate with GitHub
   - Select your repository: `BahnaGroup/Flash-Brain`

3. **Create a Trigger**
   - Name: `deploy-flashbrain-agents`
   - Event: **Push to branch**
   - Branch: `^main$` (or `^master$`)
   - Configuration: **Cloud Build configuration file**
   - Location: `cloudbuild.yaml`

### Step 2: Set Up Secrets (One-Time)

```bash
# Create secrets in Secret Manager
gcloud secrets create gemini-api-key --data-file=- <<< 'your-api-key'
gcloud secrets create supabase-url --data-file=- <<< 'your-supabase-url'
gcloud secrets create supabase-key --data-file=- <<< 'your-supabase-key'

# Grant Cloud Build access to secrets
PROJECT_NUMBER=$(gcloud projects describe flash-brain-478019 --format='value(projectNumber)')

gcloud secrets add-iam-policy-binding gemini-api-key \
  --member="serviceAccount:${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"

gcloud secrets add-iam-policy-binding supabase-url \
  --member="serviceAccount:${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"

gcloud secrets add-iam-policy-binding supabase-key \
  --member="serviceAccount:${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

### Step 3: Deploy

#### Option A: Automatic (Push to GitHub)
```bash
git add .
git commit -m "Setup Cloud Build deployment"
git push origin main
```

The build will automatically trigger and deploy all 4 services!

#### Option B: Manual Trigger (From Console)
1. Go to Cloud Build > Triggers
2. Find your trigger
3. Click **"Run"**

### Step 4: Monitor Build

View build progress:
- Console: https://console.cloud.google.com/cloud-build/builds
- CLI: `gcloud builds list --limit=5`

---

## Alternative: Deploy from Console (Per-Service)

If you want to deploy each service individually from the Console:

### For Each Service:

1. **Go to Cloud Run** → https://console.cloud.google.com/run
2. Click **"Create Service"**
3. Choose:
   - ✅ **Continuously deploy from repository** (recommended)
   - OR **Deploy one revision from source**

4. **Configure:**
   - Repository: `BahnaGroup/Flash-Brain`
   - Branch: `main`
   - Build type: **Dockerfile**
   - Dockerfile path:
     - Orchestrator: `Agents/FlashBrain Orchestrator/Dockerfile`
     - Project Decomp: `Agents/Project Decomp Graph/Dockerfile`
     - Freelancer: `Agents/Select Freelancer Graph/Dockerfile`
     - Summarize: `Agents/summarize/Dockerfile`

5. **Service Settings:**
   ```
   FlashBrain Orchestrator:
   - Min instances: 1
   - Max instances: 10
   - Memory: 1 GiB
   - CPU: 1
   
   Other Services:
   - Min instances: 0
   - Max instances: 5
   - Memory: 512 MiB
   - CPU: 1
   ```

6. **Environment Variables** (Add in Console):
   - Use Secret Manager for sensitive data
   - Or add directly: `GOOGLE_API_KEY`, `SUPABASE_URL`, etc.

---

## Comparison

| Method | Pros | Cons |
|--------|------|------|
| **Cloud Build + GitHub** | ✅ Automated<br>✅ CI/CD<br>✅ Deploys all services<br>✅ Version controlled | Requires `cloudbuild.yaml` |
| **Console (Continuous)** | ✅ Visual UI<br>✅ Easy setup | Manual for each service<br>Need to link 4 repos |
| **Local Script** | ✅ Quick testing<br>✅ One command | Requires local gcloud CLI |

## Recommended Approach

**Use Cloud Build + GitHub** for production:
1. Push `cloudbuild.yaml` to your repo
2. Set up trigger once
3. Future deployments = just `git push`

This gives you:
- ✅ Automatic deployments on every push
- ✅ Build history and rollback
- ✅ Proper minimum instances configuration
- ✅ No local dependencies

---

## Next Steps

1. Commit `cloudbuild.yaml` to your repo
2. Set up GitHub trigger in Cloud Build
3. Configure secrets in Secret Manager
4. Push to GitHub → automatic deployment! 🚀
