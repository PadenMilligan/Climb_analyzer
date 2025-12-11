# 🚀 Deploying Climb Analyzer to Render

## Prerequisites

1. Create a free account at [render.com](https://render.com)
2. Connect your GitHub account to Render
3. Push your code to a GitHub repository

## Deployment Steps

### Option 1: Using render.yaml (Recommended)

1. **Make sure `render.yaml` is in your repository root**
2. **On Render Dashboard:**
   - Click "New +" → "Blueprint"
   - Connect your GitHub repository
   - Render will automatically detect `render.yaml` and configure the service

### Option 2: Manual Setup

1. **Create a new Web Service:**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select the repository and branch

2. **Configure the service:**
   - **Name:** `climb-analyzer` (or your preferred name)
   - **Environment:** `Python 3`
   - **Region:** Choose closest to you
   - **Branch:** `main` (or your default branch)
   - **Root Directory:** Leave empty (or `./` if needed)
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `cd climb_analyzer && python app.py`

3. **Set Environment Variables:**
   - Click "Environment" tab
   - Add: `PYTHON_VERSION` = `3.10.12` (or `3.11.x`)
   - Add: `PORT` = `5000` (Render will set this automatically, but good to have)

4. **Update app.py for Render:**
   - Render provides the PORT via environment variable
   - The app should use: `os.environ.get('PORT', 5000)`

5. **Click "Create Web Service"**

## Important Notes

### MediaPipe Compatibility

MediaPipe should work on Render's Linux environment, but if you encounter issues:

1. **Check Python Version:**
   - MediaPipe requires Python 3.8-3.11
   - Set `PYTHON_VERSION` environment variable to `3.10.12` or `3.11.x`

2. **If MediaPipe still fails:**
   - Try removing the version constraint in `requirements.txt`
   - MediaPipe will install the latest compatible version

### App Configuration for Render

The app needs to use the PORT environment variable. Update `app.py`:

```python
import os

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
```

### File Storage

⚠️ **Important:** Render's free tier has **ephemeral storage**. Files uploaded will be deleted when the service restarts.

**Solutions:**
1. **Use Render Disk (Paid):** Add persistent disk storage
2. **Use Cloud Storage:** Integrate with AWS S3, Google Cloud Storage, or similar
3. **Process and delete:** Process videos immediately and don't store them

### Performance Considerations

- Video processing is CPU-intensive
- Free tier has limited resources
- Consider upgrading for better performance
- Large videos may timeout on free tier

## Troubleshooting

### Build Fails with MediaPipe Error

1. Check Python version (should be 3.8-3.11)
2. Try removing version constraints: `mediapipe` instead of `mediapipe>=0.10.0`
3. Check Render's build logs for specific error messages

### App Crashes on Startup

1. Check logs in Render dashboard
2. Verify all dependencies are in `requirements.txt`
3. Ensure start command is correct: `cd climb_analyzer && python app.py`

### Videos Not Processing

1. Check file size limits (Render free tier has limits)
2. Check timeout settings
3. Verify MediaPipe is working (check logs)

## Alternative: Railway.app

If Render continues to have issues, try [Railway.app](https://railway.app):

1. Sign up and connect GitHub
2. Deploy from repository
3. Railway auto-detects Python apps
4. Add environment variables as needed

Railway often has better compatibility with MediaPipe.

