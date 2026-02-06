# 🎯 Amazon Alexa Sentiment Analysis

<div align="center">

![Python](https://img.shields.io/badge/Python-3.13+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-spaCy-09A3D5?style=for-the-badge&logo=spacy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

**A Natural Language Processing project for sentiment classification of Amazon Alexa product reviews**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Project Structure](#-project-structure) • [Contributing](#-contributing)

</div>

---

## 📖 Overview

This project implements a **sentiment analysis pipeline** for Amazon Alexa product reviews. Using NLP techniques and the spaCy library, the system classifies customer reviews as either **positive** or **negative** based on their rating and text content.

### Key Highlights
- 🔍 **Text preprocessing** with lemmatization and stopword removal
- 📊 **3,150+ reviews** from Amazon Alexa customers
- 🏷️ **Binary sentiment classification** (Positive: rating > 3, Negative: rating ≤ 3)
- ⚡ **Custom tokenizer** using spaCy's English language model

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Data Preprocessing** | Clean and prepare raw review text for analysis |
| **Custom Tokenization** | Advanced tokenizer with lemmatization support |
| **Stopword Removal** | Filter out common English stopwords |
| **Punctuation Handling** | Remove punctuation for cleaner text analysis |
| **Sentiment Labeling** | Convert 1-5 star ratings to binary sentiment labels |

---

## 🛠️ Installation

### Prerequisites
- Python 3.13 or higher
- pip (Python package manager)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/MustafaKocamann/NLP-Projects.git
   cd NLP-Projects
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install numpy pandas spacy jupyter
   ```

4. **Download spaCy English model**
   ```bash
   python -m spacy download en_core_web_sm
   ```

---

## 🚀 Usage

### Running the Notebook

1. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

2. Open `sentiment_analysis.ipynb`

3. Run all cells to execute the sentiment analysis pipeline

### Using the Custom Tokenizer

```python
from tokenizer_input import CustomTokenizerExample

# Initialize tokenizer
tokenizer = CustomTokenizerExample()

# Clean and tokenize text
text = "Those were the best days of my life!"
tokens = tokenizer.text_data_cleaning(text)
print(tokens)  # Output: ['good', 'day', 'life']
```

---

## 📁 Project Structure

```
sentiment-analysis/
│
├── 📓 sentiment_analysis.ipynb   # Main analysis notebook
├── 🐍 tokenizer_input.py         # Custom tokenizer module
├── 📄 README.md                  # Project documentation
├── 📝 .gitignore                 # Git ignore file
└── 📁 venv/                      # Virtual environment (not tracked)
```

---

## 📊 Dataset

The project uses the **Amazon Alexa Reviews Dataset** containing:

| Column | Description |
|--------|-------------|
| `rating` | Customer rating (1-5 stars) |
| `date` | Review date |
| `variation` | Product variation (e.g., Charcoal Fabric, Black Dot) |
| `verified_reviews` | Customer review text |
| `feedback` | Binary feedback indicator |

### Sentiment Distribution
- **Positive reviews (rating > 3):** 2,741 (87%)
- **Negative reviews (rating ≤ 3):** 409 (13%)

> ⚠️ **Note:** The dataset is not included in this repository. You can download it from [Kaggle](https://www.kaggle.com/datasets/sid321axn/amazon-alexa-reviews).

---

## 🔧 Technical Details

### Text Processing Pipeline

1. **Tokenization:** Split text into individual tokens using spaCy
2. **Lemmatization:** Convert words to their base form
3. **Lowercase Conversion:** Normalize text to lowercase
4. **Stopword Removal:** Remove common English stopwords
5. **Punctuation Removal:** Clean punctuation from tokens

### Technologies Used

- **[spaCy](https://spacy.io/)** - Industrial-strength NLP library
- **[Pandas](https://pandas.pydata.org/)** - Data manipulation and analysis
- **[NumPy](https://numpy.org/)** - Numerical computing
- **[Jupyter](https://jupyter.org/)** - Interactive notebooks

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

## 👤 Author

**Mustafa Kocaman**

- GitHub: [@MustafaKocamann](https://github.com/MustafaKocamann)
- Email: mustafakocaman789@gmail.com

---

<div align="center">

⭐ **If you found this project helpful, please give it a star!** ⭐

</div>
