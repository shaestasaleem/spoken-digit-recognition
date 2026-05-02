# 🚀 QUICK START GUIDE - PhonemeIQ v3.0

## ⚡ 5-Minute Setup (Windows/Mac/Linux)

### Step 1: Install Python
- Download from [python.org](https://python.org) (3.8+)
- Make sure to check "Add Python to PATH" during installation

### Step 2: Open Terminal/Command Prompt
Navigate to the project folder:
```bash
cd speech_gui
```

### Step 3: Run Setup Script

**Windows:**
```bash
setup.bat
```

**Mac/Linux:**
```bash
chmod +x setup.sh
./setup.sh
```

### Step 4: Start the App
```bash
streamlit run app_professional.py
```

Your browser will open automatically at `http://localhost:8501`

---

## 🎯 Using the App

### Inference Tab
1. **Select Model**: Choose SVM, CNN, or Both
2. **Upload Audio**: Click "Upload File" and select WAV/MP3/OGG
3. **Run**: Click the "Run Inference" button
4. **View Results**: See predictions with confidence scores

### Analytics Tab
- View model performance metrics
- Compare SVM vs CNN accuracy
- Check feature engineering details

### Documentation Tab
- Read complete technical documentation
- Learn about the models and dataset
- Understand the system architecture

### Settings Tab
- Check model status
- View system information
- Configure audio parameters

---

## 🐳 Using Docker (Alternative)

```bash
docker-compose up
```

Opens at `http://localhost:8501`

---

## ☁️ Deploy to Streamlit Cloud (1 minute!)

1. Push to GitHub:
```bash
git add .
git commit -m "Initial commit"
git push
```

2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New App" → Select your repo → Set main file to `app_professional.py`
4. Done! Your app is live 🎉

---

## 📊 Test Files

Use any of these to test:
- Pre-recorded: `recorded_digit.wav`
- Training data: `Spoken Digit Recognition (0-9)/recordings/`

---

## ❓ Troubleshooting

**"ModuleNotFoundError"?**
```bash
pip install -r requirements.txt
```

**Port already in use?**
```bash
streamlit run app_professional.py --server.port 8502
```

**Models not loading?**
- Make sure these files exist in the same folder:
  - `cnn_model.h5`
  - `svm_model.pkl`
  - `scaler.pkl`
  - `label_encoder.pkl`

---

## 📞 Need Help?

Check these files in order:
1. This file (QUICK_START.md)
2. README.md (detailed info)
3. DEPLOYMENT_GUIDE.md (for deployment)

---

**That's it! Enjoy PhonemeIQ! 🎙️**
