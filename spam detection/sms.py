# Importing the libraries
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import pickle
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the dataset
df = pd.read_csv("spam.csv", encoding = "latin-1")

# Drop unnecessary columns
df = df[["v1", "v2"]]
df.columns = ["label", "message"]

# Data Cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_message"] = df["message"].apply(clean_text)
print(df.head())

# Encoding
df["label"] = df["label"].map({"ham": 0, "spam": 1})
print(df.head())

X = df["clean_message"]
y = df["label"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")

tfidf = TfidfVectorizer()

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Logistic Regression
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter = 500)

# Training
lr.fit(X_train_tfidf, y_train)

# Prediction
lrpred = lr.predict(X_test_tfidf)

# Metrics
print(f"Accuracy: {accuracy_score(y_test, lrpred)}")
print(f"F1 Score: {f1_score(y_test, lrpred)}")
print(classification_report(y_test, lrpred))

# Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)


X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Max Lenght
max_len = max(len(seq) for seq in X_train_seq)
print(max_len)

# Padding
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

# Word Count
vocab_size = len(tokenizer.word_index) + 1

# LSTM Model
model = Sequential([
    Embedding(input_dim = vocab_size, output_dim= 128, input_length=max_len),
    LSTM(128),
    Dropout(0.3),
    Dense(1, activation = "sigmoid")

])

# Compile
model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

# Training
history = model.fit(
    X_train_pad,
    y_train,
    epochs = 2,
    batch_size = 32,
    validation_split = 0.1,
    verbose = 1
)

# Prediction
lstm_pred = (model.predict(X_test_pad) > 0.5).astype("int")

acc_lstm = accuracy_score(y_test, lstm_pred)
f1_lstm = f1_score(y_test, lstm_pred)

# Metrics
print(f"Accuracy: {acc_lstm: .2f}")
print(f"F1 Score: {f1_lstm: .2f}")
print(classification_report(y_test, lstm_pred))

# Confusion Matrix
cm_lr = confusion_matrix(y_test, lrpred)
cm_lstm = confusion_matrix(y_test, lstm_pred)

plt.figure(figsize=(10,5))

# Logistic Regression Confusion Matrix
plt.subplot(1,2,1)
sns.heatmap(cm_lr, annot = True, fmt = "d", cmap = "Blues")
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")

# LSTM Confusion Matrix
plt.subplot(1,2,2)
sns.heatmap(cm_lstm, annot = True, fmt = "d", cmap = "Greens")
plt.title("Confusion Matrix - LSTM")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.show()

# Save Model
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer,f)

model.save("spam_detection_model.h5")

# Save Logistic Regression Model
with open("lr_model.pkl", "wb") as f:
    pickle.dump(lr, f)

# Save TF-IDF Vectorizer
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

print("All models saved successfully!")
