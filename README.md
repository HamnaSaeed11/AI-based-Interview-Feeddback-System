# 🎯 AI-Based Interview Feedback System  

An AI-powered web application that analyzes **facial expressions and voice tone** during interviews to provide **real-time feedback and performance insights**.

---

## 🚀 Features  

- 🎥 Facial Emotion Recognition (Webcam)  
- 🎙️ Speech Emotion Analysis (Microphone)  
- 📊 Visualization Dashboard (Emotion insights & charts)  
- 🔐 User Authentication (Login & Signup)  
- ⚡ Real-time Processing  

---

## 🧠 Technologies Used  

### 🔹 Programming & Framework  
- Python  
- Flask  

### 🔹 Deep Learning Models  
- EfficientNet-B2 (CNN) – Facial Emotion Recognition  
- CNN + BiLSTM – Speech Emotion Recognition  

### 🔹 Libraries  
- OpenCV (video processing)  
- Librosa (audio feature extraction - MFCC)  
- NumPy, Pandas (data handling)  
- Matplotlib, Seaborn (visualization)  
- Scikit-learn (evaluation metrics)  

---

## 📂 Datasets  

- FER-2013 (Facial Emotion Dataset)  
- RAVDESS (Speech Emotion Dataset)  
- CREMA-D (Audio Emotion Dataset)  

---

## ⚙️ How It Works  

1. User logs into the system  
2. Selects interview mode (Video / Audio)  
3. System captures input via webcam or microphone  
4. Deep learning models analyze emotions  
5. Results are displayed with visual insights  

---

## 💻 Installation  

```bash
# Clone repository
git clone https://github.com/your-username/your-repo-name.git

# Navigate to project folder
cd your-repo-name

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py

Open in browser:
http://127.0.0.1:5000/
