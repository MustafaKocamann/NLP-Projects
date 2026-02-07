from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import uvicorn

from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re

app = FastAPI(
    title = "Spam Detection API",
    description = "API for spam detection",
    version = "1.0.0"
)

# Load Model
model = tf.keras.models.load_model("spam_detection_model.h5")

# Load Tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_len = model.input_shape[1]

# Preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Request 
class SMSRequest(BaseModel):
    message: str

# Response
class SMSResponse(BaseModel):
    label: str
    probability: float

# Prediction
@app.post("/predict", response_model=SMSResponse)
def predict(request: SMSRequest):
    raw_message = request.message
    cleaned_text = clean_text(raw_message)
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(sequence, maxlen=max_len)
    prob = float(model.predict(padded)[0][0])
    label = "spam" if prob > 0.5 else "ham"
    return SMSResponse(label=label, probability=prob)

if __name__ == "__main__":
    uvicorn.run("api:app", host = "127.0.0.1", port = 8000, reload = True)
    
    