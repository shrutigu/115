import tensorflow.keras
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import tensorflow as tf
import pandas as pd

review = ['you can try for better camera options as well, but for this amount, its great to buy']
train_data = pd.read_csv("updated_product_dataset.csv")
training_sentences = []

for i in range(len(train_data)):
    sentence = train_data.loc[i, "Text"]
    training_sentences.append(sentence)

model = load_model("Customer_Review_Text_Emotion.h5")

vocab_size = 40000
max_length = 100
trunc_type = "post"
padding_type = "post"
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

# dictionary where key : emotion , value
encode_emotions = {"Neutral": 0, "Positive": 1, "Negative": 2}

sequences = tokenizer.texts_to_sequences(review)
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
result = model.predict(padded)
label = np.argmax(result , axis=1)
print(label)

for emotion in encode_emotions:
    if encode_emotions[emotion]  ==  label:
        print(emotion)