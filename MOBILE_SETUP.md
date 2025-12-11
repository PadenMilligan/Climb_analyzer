# 📱 Using Climb Analyzer on Your iPhone

## Quick Start (Local Network - Easiest)

### Option 1: Using the Helper Script (Recommended)

1. **On your computer:**
   ```bash
   cd climb_analyzer
   python start_server.py
   ```

2. **The script will show you:**
   - Your computer's IP address (e.g., `192.168.1.100`)
   - The URL to use on your iPhone

3. **On your iPhone:**
   - Make sure your iPhone is on the **SAME WiFi network** as your computer
   - Open Safari
   - Go to the URL shown (e.g., `http://192.168.1.100:5000`)
   - You can now upload and analyze videos!

### Option 2: Manual Setup

1. **Find your computer's IP address:**
   - **Windows:** Open Command Prompt, type `ipconfig`, look for "IPv4 Address"
   - **Mac/Linux:** Open Terminal, type `ifconfig` or `ip addr`, look for your WiFi adapter's IP

2. **Start the server:**
   ```bash
   cd climb_analyzer
   python app.py
   ```

3. **On your iPhone:**
   - Connect to the same WiFi network
   - Open Safari
   - Go to: `http://YOUR_IP_ADDRESS:5000` (replace with your actual IP)

---

## 🚀 Cloud Deployment (Access from Anywhere)

For permanent access from anywhere (not just same WiFi), deploy to a cloud service:

### Option A: Render (Free Tier Available)
1. Create account at [render.com](https://render.com)
2. Create a new Web Service
3. Connect your GitHub repository
4. Set build command: `pip install -r requirements.txt`
5. Set start command: `cd climb_analyzer && python app.py`
6. Deploy!

### Option B: Railway
1. Create account at [railway.app](https://railway.app)
2. Deploy from GitHub
3. Add environment variables if needed
4. Deploy!

### Option C: Heroku
1. Create account at [heroku.com](https://heroku.com)
2. Install Heroku CLI
3. Create app: `heroku create your-app-name`
4. Deploy: `git push heroku main`

---

## 📋 Requirements for Local Network Access

✅ **Both devices must be on the same WiFi network**
✅ **Computer must be running the Flask server**
✅ **Firewall may need to allow port 5000** (Windows Firewall will prompt you)

---

## 🔧 Troubleshooting

### Can't connect from iPhone?

1. **Check WiFi:** Make sure both devices are on the same network
2. **Check IP address:** Verify you're using the correct IP (not 127.0.0.1)
3. **Check firewall:** Windows Firewall may block the connection - allow it when prompted
4. **Try different browser:** Sometimes Safari works better than Chrome on iOS
5. **Check port:** Make sure nothing else is using port 5000

### Video upload is slow?

- Large video files take time to upload and process
- Consider compressing videos before uploading
- Processing time depends on video length and your computer's performance

### Server won't start?

- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Check if port 5000 is already in use
- Try a different port: Change `port=5000` to `port=5001` in `app.py`

---

## 💡 Tips for Gym Use

1. **Use your phone's hotspot** if gym WiFi is unreliable:
   - Turn on Personal Hotspot on iPhone
   - Connect your laptop to the hotspot
   - Then access from iPhone using the hotspot's network

2. **Bookmark the URL** on your iPhone for quick access

3. **Keep laptop plugged in** - video processing is CPU-intensive

4. **Pre-process videos** on your phone to reduce file size before uploading

---

## 🎯 Best Setup for Gym

**Recommended:** Use a laptop with the app running, access via iPhone on same WiFi. This gives you:
- ✅ Full processing power on laptop
- ✅ Easy access from phone
- ✅ No cloud costs
- ✅ Works offline (no internet needed)

