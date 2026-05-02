# 🎙️ PhonemeIQ v3.0 — Professional Spoken Digit Recognition System

> **Academic-Grade Speech Recognition using Advanced Machine Learning**
>
> Harvard-LUMS Institutional Standard | Speech Processing Laboratory

---

## 📋 Overview

PhonemeIQ is a professional-grade spoken digit recognition system that combines Support Vector Machine (SVM) and Convolutional Neural Network (CNN) architectures to achieve state-of-the-art accuracy in recognizing spoken English digits (0-9).

### ✨ Key Features

- **Dual-Model Architecture**: Leverages both SVM and CNN for robust predictions
- **Professional UI**: Academic-grade interface with dark academia aesthetic
- **High Accuracy**: SVM (96.33%) and CNN (97.41%) on FSDD dataset
- **Fast Inference**: Real-time digit recognition from audio files
- **Production-Ready**: Fully deployable on Streamlit Cloud, Docker, and cloud platforms
- **Comprehensive Analytics**: Model comparison, metrics, and detailed documentation
- **Flexible Input**: Support for WAV, MP3, and OGG audio formats

---

## 🎯 Model Performance

| Metric | SVM | CNN |
|--------|-----|-----|
| **Accuracy** | 96.33% | 97.41% |
| **Precision** | 96.1% | 97.2% |
| **Recall** | 96.3% | 97.4% |
| **F1-Score** | 96.2% | 97.3% |
| **Inference Time** | ~15ms | ~45ms |
| **Model Size** | ~2MB | ~12MB |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Audio files (WAV, MP3, or OGG format)

### Installation

1. **Clone or download the repository:**
   ```bash
   cd speech_gui
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify model files are present:**
   ```bash
   # Check for these files:
   # - cnn_model.h5
   # - svm_model.pkl
   # - scaler.pkl
   # - label_encoder.pkl
   ```

### Running Locally

**Using the professional version:**
```bash
streamlit run app_professional.py
```

The application will open at `http://localhost:8501`

---

## 📦 Project Structure

```
speech_gui/
├── app_professional.py           # Main professional Streamlit app
├── cnn_model.h5                  # Pre-trained CNN model (Keras)
├── svm_model.pkl                 # Pre-trained SVM model (scikit-learn)
├── scaler.pkl                    # Feature scaler for SVM
├── label_encoder.pkl             # Label encoder for digit classes
├── requirements.txt              # Python dependencies
├── .streamlit/
│   └── config.toml              # Streamlit configuration
├── Spoken Digit Recognition (0-9)/
│   └── recordings/              # Training dataset (FSDD)
├── Spoken_Digit_Recognition_Final (2).ipynb  # Training notebook
└── README.md                     # This file
```

---

## 🏗️ Technical Architecture

### Data Pipeline

```
Audio Input (WAV/MP3/OGG)
    ↓
Preprocessing (Resample to 8kHz)
    ├→ SVM Path: MFCC + ZCR + Energy (15 features)
    └→ CNN Path: Mel Spectrogram (64×64×1)
    ↓
Model Inference
    ├→ SVM: RBF Kernel Classification
    └→ CNN: 3-Layer Convolutional Network
    ↓
Prediction + Confidence Score
    ↓
Result Visualization
```

### Feature Engineering

**SVM Features (15-dimensional):**
- MFCC Coefficients: 13 coefficients capturing frequency characteristics
- Zero Crossing Rate: 1 feature measuring signal oscillations
- Energy: 1 feature measuring signal power

**CNN Features (64×64×1 Mel Spectrogram):**
- Time-frequency representation using 64 mel-frequency bins
- Frequency range optimized for speech: 0-4000 Hz
- Log-scaled power output for better neural network learning
- Min-Max normalization applied for stability

### Model Details

**SVM Configuration:**
```python
- Kernel: RBF (Radial Basis Function)
- C: 100 (regularization parameter)
- Gamma: 0.001 (RBF kernel parameter)
- Classes: 10 (digits 0-9)
```

**CNN Architecture:**
```
Input Layer: (64, 64, 1)
    ↓
Conv2D (32 filters, 3×3, ReLU) + MaxPooling2D (2×2)
    ↓
Conv2D (64 filters, 3×3, ReLU) + MaxPooling2D (2×2)
    ↓
Flatten
    ↓
Dense (128 units, ReLU) + Dropout (0.5)
    ↓
Output Dense (10 units, Softmax)
```

---

## 🎨 User Interface Sections

### 1. **Inference Tab** 🎯
- Model selection (SVM / CNN / Both)
- Audio input methods (Upload file or record)
- Real-time inference with confidence scores
- Detailed audio information display
- Project overview and team information

### 2. **Analytics Tab** 📊
- Model performance metrics
- Accuracy comparison charts
- Feature engineering details
- Performance benchmarks
- Cross-model analysis

### 3. **Documentation Tab** 📚
- Complete technical documentation
- System architecture explanation
- Data pipeline description
- Model configuration details
- Usage guidelines and limitations

### 4. **Settings Tab** ⚙️
- Model status indicators
- Audio configuration parameters
- System information
- Version details

---

## 🌐 Deployment Guide

### Option 1: Streamlit Cloud (Recommended)

**Easiest deployment option - completely free!**

1. **Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/speech_gui.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New App"
   - Select your GitHub repository
   - Set main file to: `app_professional.py`
   - Click "Deploy"

3. **Your app is live!** Share the link with anyone

### Option 2: Docker (For Custom Servers)

**Dockerfile:**
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app_professional.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**Build and run:**
```bash
docker build -t phonemeiq .
docker run -p 8501:8501 phonemeiq
```

### Option 3: Heroku Deployment

**Create Procfile:**
```
web: streamlit run app_professional.py --server.port=$PORT --server.address=0.0.0.0
```

**Deploy:**
```bash
heroku create your-app-name
git push heroku main
```

### Option 4: AWS EC2

1. Launch Ubuntu instance
2. Install Python 3.10+
3. Clone repository
4. Install dependencies
5. Run: `streamlit run app_professional.py --server.port 80`
6. Use nginx as reverse proxy

### Option 5: Google Cloud Run

```bash
# Create app.yaml
gcloud run deploy phonemeiq \
  --source . \
  --platform managed \
  --region us-central1
```

---

## 🔧 Configuration

### Streamlit Configuration (`.streamlit/config.toml`)

Customize theme colors, server settings, and client behavior:

```toml
[theme]
primaryColor = "#b8972a"      # Gold accent
backgroundColor = "#faf8f4"    # Cream background
secondaryBackgroundColor = "#f5f2ec"
textColor = "#0d0d0d"         # Dark ink

[server]
maxUploadSize = 100           # Max 100MB files
enableXsrfProtection = true   # Security
enableCORS = false            # Security

[client]
showErrorDetails = true       # For debugging
```

### Environment Variables (Optional)

Create `.env` file for sensitive configuration:

```env
LOG_LEVEL=INFO
DEBUG=False
MAX_UPLOAD_SIZE=100
```

---

## 📊 Dataset Information

**Free Spoken Digit Dataset (FSDD)**

- **Total Samples**: 3,000+ audio files
- **Classes**: 10 (digits 0-9)
- **Speakers**: 6 unique speakers
- **Samples per Digit/Speaker**: 50 recordings
- **Audio Format**: WAV, 16-bit, Mono
- **Sample Rate**: 8kHz (native)
- **Average Duration**: 0.5-2 seconds per clip
- **Language**: English
- **Quality**: Clean, minimal background noise

**Training Split:**
- Training: 70% (2,100 samples)
- Validation: 15% (450 samples)
- Testing: 15% (450 samples)

---

## 🚨 Troubleshooting

### Issue: Model files not found
**Solution:** Ensure these files are in the same directory as app:
- `cnn_model.h5`
- `svm_model.pkl`
- `scaler.pkl`
- `label_encoder.pkl`

### Issue: "ModuleNotFoundError: No module named 'tensorflow'"
**Solution:** Install dependencies:
```bash
pip install -r requirements.txt
```

### Issue: Audio processing error
**Solution:** Try resampling audio to 16-bit mono WAV:
```bash
ffmpeg -i input.mp3 -acodec pcm_s16le -ar 8000 -ac 1 output.wav
```

### Issue: Streamlit Cloud timeout
**Solution:** Add to `.streamlit/config.toml`:
```toml
[client]
toolbarMode = "minimal"

[server]
runOnSave = false
```

---

## 📈 Performance Optimization

### For Faster Inference:
1. Use SVM model only (15ms vs 45ms for CNN)
2. Keep audio duration under 2 seconds
3. Use pre-processed audio (already 8kHz mono)

### For Better Accuracy:
1. Use CNN model (97.41% vs 96.33%)
2. Use "Both" mode and ensemble results
3. Ensure clean audio input (minimal background noise)

---

## 🔐 Security Considerations

- ✅ CSRF protection enabled
- ✅ File upload size limited to 100MB
- ✅ No sensitive data storage
- ✅ Model files are read-only
- ✅ Input validation for audio files

---

## 📝 API Usage (Advanced)

If you want to use the models programmatically:

```python
import librosa
import pickle
from tensorflow.keras.models import load_model

# Load models
svm_model = pickle.load(open("svm_model.pkl", "rb"))
cnn_model = load_model("cnn_model.h5")
scaler = pickle.load(open("scaler.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

# Load audio
signal, sr = librosa.load("audio.wav", sr=None)

# Extract features (see app_professional.py for functions)
features = extract_features_svm(signal, sr).reshape(1, -1)
features_scaled = scaler.transform(features)

# Predict
prediction = svm_model.predict(features_scaled)[0]
digit = le.inverse_transform([prediction])[0]
print(f"Predicted digit: {digit}")
```

---

## 📚 References & Citations

- **FSDD Dataset**: [github.com/Jakobovski/free-spoken-digit-dataset](https://github.com/Jakobovski/free-spoken-digit-dataset)
- **Librosa**: [librosa.org](https://librosa.org)
- **TensorFlow/Keras**: [tensorflow.org](https://tensorflow.org)
- **scikit-learn**: [scikit-learn.org](https://scikit-learn.org)
- **Streamlit**: [streamlit.io](https://streamlit.io)

---

## 👥 Team

**Speech Processing Laboratory - Harvard-LUMS**

- Savaira Majeed
- Shaesta Saleem
- Mustajab Zahra
- Faizan Nazik
- Sarfraz Ahmad
- Muhammad Gulfam

---

## 📄 License

This project is provided for educational and research purposes. The pre-trained models and dataset follow their respective licenses.

---

## 📞 Support & Feedback

For issues, suggestions, or contributions:

1. Check the troubleshooting section above
2. Review the Documentation tab in the app
3. Check console output for detailed error messages
4. Ensure all dependencies are correctly installed

---

## 🎯 Future Enhancements

- [ ] Real-time audio recording support
- [ ] Batch processing of multiple files
- [ ] Model retraining with custom data
- [ ] Advanced audio preprocessing options
- [ ] API endpoint for programmatic access
- [ ] Multi-language support
- [ ] Confidence threshold customization
- [ ] Audio visualization (waveform, spectrogram)
- [ ] Model explainability (SHAP, LIME)
- [ ] Export predictions to CSV/JSON

---

**PhonemeIQ v3.0** | *Professional Speech Recognition* | *2025*

*Designed and developed for academic excellence and production-grade deployment.*
