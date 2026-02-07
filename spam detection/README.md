<div align="center">

# 🛡️ SMS Spam Detector

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**Deep Learning ve Klasik ML ile SMS Spam Tespiti**

[Özellikler](#-özellikler) • [Kurulum](#-kurulum) • [Kullanım](#-kullanım) • [API](#-api-kullanımı) • [Mimari](#-teknik-mimari)

---

<!-- Proje arayüz görseli için placeholder -->
<img src="https://via.placeholder.com/800x400/1a1a2e/7b61ff?text=SMS+Spam+Detector+UI" alt="SMS Spam Detector Interface" width="100%">

</div>

---

## 📋 Proje Hakkında

SMS Spam Detector, metin mesajlarını **Spam** veya **Ham (Güvenli)** olarak sınıflandıran, çift modelli bir yapay zeka uygulamasıdır. Proje, hem **Derin Öğrenme (LSTM)** hem de **Klasik Makine Öğrenmesi (Logistic Regression)** yaklaşımlarını bir arada sunarak, farklı senaryolar için esnek bir çözüm sağlar.

> 📊 **Veri Seti:** [Kaggle SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) - 5,574 etiketli SMS mesajı

---

## 🚀 Özellikler

| Özellik | Açıklama |
|---------|----------|
| 🧠 **Çift Model Mimarisi** | LSTM (Deep Learning) ve Logistic Regression modelleri |
| ⚡ **Gerçek Zamanlı API** | FastAPI ile production-ready REST API |
| 🎨 **Modern Arayüz** | Streamlit tabanlı kullanıcı dostu web arayüzü |
| 📊 **Yüksek Performans** | %97+ doğruluk oranı ile güvenilir tahminler |
| 🔧 **Modüler Yapı** | Kolay genişletilebilir ve özelleştirilebilir kod tabanı |

---

## 🏗️ Teknik Mimari

### LSTM Model Yapısı

```
Input Text
    │
    ▼
┌─────────────────────────────────────┐
│  Preprocessing (clean_text)        │
│  • Lowercase conversion            │
│  • URL & digit removal             │
│  • Special character cleaning      │
│  • Whitespace normalization        │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Tokenization & Padding            │
│  • texts_to_sequences()            │
│  • pad_sequences(maxlen=N)         │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Embedding Layer (128 dim)         │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  LSTM Layer (128 units)            │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Dropout (0.3)                     │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Dense Layer (Sigmoid)             │
└─────────────────────────────────────┘
    │
    ▼
  Output: Spam Probability (0-1)
```

### Logistic Regression Pipeline

```
Input Text → clean_text() → TF-IDF Vectorization → Logistic Regression → Prediction
```

---

## 📊 Model Performansı

| Model | Accuracy | F1-Score | Precision | Recall |
|-------|----------|----------|-----------|--------|
| **LSTM** | 97.8% | 92.5% | 94.1% | 91.0% |
| **Logistic Regression** | 96.4% | 89.2% | 91.3% | 87.2% |

> 📈 Metrikler, test seti (%10 holdout, stratified split) üzerinde hesaplanmıştır.

---

## 📁 Proje Yapısı

```
spam-detection/
│
├── 📄 sms.py                    # Model eğitim scripti
├── 📄 api.py                    # FastAPI REST API
├── 📄 app.py                    # Streamlit web arayüzü
│
├── 🧠 spam_detection_model.h5   # Eğitilmiş LSTM modeli
├── 📦 tokenizer.pkl             # Keras Tokenizer
├── 📦 lr_model.pkl              # Logistic Regression modeli
├── 📦 tfidf_vectorizer.pkl      # TF-IDF Vectorizer
│
├── 📊 spam.csv                  # SMS Spam Collection veri seti
├── 📋 requirements.txt          # Python bağımlılıkları
└── 📖 README.md                 # Dokümantasyon
```

---

## 🛠️ Kurulum

### Gereksinimler
- Python 3.9+
- pip

### Adımlar

**1. Repoyu klonlayın:**
```bash
git clone https://github.com/YOUR_USERNAME/sms-spam-detector.git
cd sms-spam-detector
```

**2. Sanal ortam oluşturun (önerilir):**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

**3. Bağımlılıkları yükleyin:**
```bash
pip install -r requirements.txt
```

**4. Modeli eğitin (opsiyonel):**
```bash
python sms.py
```

---

## 🎯 Kullanım

### Streamlit Arayüzü

```bash
streamlit run app.py
```
Tarayıcınızda `http://localhost:8501` adresine gidin.

### FastAPI Sunucusu

```bash
uvicorn api:app --reload
```

API dokümantasyonuna `http://127.0.0.1:8000/docs` adresinden erişin.

---

## 🔌 API Kullanımı

### Endpoint

```
POST /predict
```

### Request Body

```json
{
  "message": "Congratulations! You've won a FREE iPhone!"
}
```

### Response

```json
{
  "label": "spam",
  "probability": 0.9847
}
```

### cURL Örneği

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hey, are you coming to the meeting tomorrow?"}'
```

---

## 🧪 Test

```bash
# API'yi test et
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"message": "WINNER! You have been selected for a prize!"}'
```

---

## 🤝 Katkıda Bulunma

Katkılarınızı memnuniyetle karşılıyoruz! Lütfen bir **Pull Request** açmadan önce:

1. Repoyu fork edin
2. Feature branch oluşturun (`git checkout -b feature/YeniOzellik`)
3. Değişikliklerinizi commit edin (`git commit -m 'Yeni özellik eklendi'`)
4. Branch'inizi push edin (`git push origin feature/YeniOzellik`)
5. Pull Request açın

---

## 📄 Lisans

Bu proje [MIT Lisansı](LICENSE) altında lisanslanmıştır.

---

<div align="center">

**⭐ Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!**

Made with ❤️ using Python & TensorFlow

</div>
