# 🎤 Spoken Digit Recognition — GUI

Speech Processing Final Lab Project 

## How to Run

### Step 1: Save Models from Notebook
Run these cells at the END of your Colab notebook:

```python
import pickle, shutil

# Save SVM model
with open('svm_model.pkl', 'wb') as f:
    pickle.dump(svm_model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

# Save CNN model
cnn_model.save('cnn_model.h5')

# Copy to Drive
shutil.copy('svm_model.pkl', '/content/drive/MyDrive/Spoken Digit Recognition (0-9)/')
shutil.copy('scaler.pkl',    '/content/drive/MyDrive/Spoken Digit Recognition (0-9)/')
shutil.copy('label_encoder.pkl', '/content/drive/MyDrive/Spoken Digit Recognition (0-9)/')
shutil.copy('cnn_model.h5',  '/content/drive/MyDrive/Spoken Digit Recognition (0-9)/')
print("✅ All models saved!")
```

### Step 2: Put These Files Together
```
your_folder/
├── app.py
├── requirements.txt
├── svm_model.pkl
├── scaler.pkl
├── label_encoder.pkl
└── cnn_model.h5
```

### Step 3: Install and Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Step 4: Open in Browser
Go to: http://localhost:8501

## Features
- Upload .wav / .ogg / .mp3 file → predict digit
- Record from microphone → predict digit
- Choose SVM, CNN, or Both models
- See confidence score for each prediction

