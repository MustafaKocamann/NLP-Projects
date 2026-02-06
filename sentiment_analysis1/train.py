import pandas as pd
import numpy as np
import re
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Veriyi Yüklemek
df = pd.read_csv("Hotel_Reviews.csv")
print(df.head())
print(df.columns)

# Label kuralı
def extract_review_and_label(row):
    neg = row["Negative_Review"].strip()
    pos = row["Positive_Review"].strip()
    
    if neg != "No Negative":
        return neg, 0

    if pos != "No Positive":
        return pos, 1

    return pos, 1

df["review"], df["label"] = zip(*df.apply(extract_review_and_label, axis=1))

print(df.head())
print(df["label"].value_counts())

# Veri Temizleme
def clean_text(text):
    text = text.lower() # küçük harfe çevirmek
    text = re.sub(r"https?://\S+", "", text) # URL'leri silmek
    text = re.sub(r"www\.\S+", "", text) # URL'leri silmek
    text = re.sub(r"<.*?>", "", text) # HTML etiketlerini silmek
    text = re.sub(r"[^\w\s]", "", text) # noktalama işaretlerini silmek
    text = re.sub(r"\d+", "", text) # sayıları silmek
    text = re.sub(r"\s+", " ", text).strip() # boşlukları silmek
    return text

df["cleaned_review"] = df["review"].apply(clean_text)

print(df.head(10))

# Veriyi bölmek
X = df["cleaned_review"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, stratify=y)

print(X_train.shape)
print(X_test.shape)

# Decision Tree Modelini Eğitmek
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

model = DecisionTreeClassifier(max_depth=15)
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Confusion Matrix:", confusion_matrix(y_test, y_pred))

# GRU Modeli Oluşturmak
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

max_len = max(len(seq) for seq in X_train_seq) # en uzun yorumun uzunluğunu bulmak için

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

vocab_size = len(tokenizer.word_index) + 1

model_gru = Sequential(
    [
        Embedding(vocab_size, 128, input_length= max_len),
        GRU(128),
        Dropout(0.5), # overfittingi önlemek için
        Dense(1, activation="sigmoid") # binary classification için sigmoid
    ]
)


# Compile etme
model_gru.compile(
    optimizer = "adam",
    loss = "binary_crossentropy",
    metrics = ["accuracy"]
)

model_gru.fit(
    X_train_pad,
    y_train,
    epochs = 1,
    batch_size = 128,
    validation_split = 0.1,
    verbose = 1
)

pred_gru = (model_gru.predict(X_test_pad) > 0.5).astype(int)

print("GRU Accuracy:", accuracy_score(y_test, pred_gru))
print("GRU F1 Score:", f1_score(y_test, pred_gru))
print("GRU Confusion Matrix:", confusion_matrix(y_test, pred_gru))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot= True, fmt = "d", cmap = "Blues")
plt.title("Confusion Matrix - GRU")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()