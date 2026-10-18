"""
Train / Retrain Chatbot Intent Classification Model
===================================================
Reads `intents.json`, builds token vocabulary and class lists,
trains a Keras intent classification neural network, and saves
`chatbot_model.h5`, `words.pkl`, and `classes.pkl` directly to `models/chatbot/`.

Usage:
    python train_chatbot.py
"""

import json
import os
import pickle
import random
from pathlib import Path

import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD

for pkg in ("punkt", "punkt_tab", "wordnet", "omw-1.4"):
    try:
        nltk.data.find(f"tokenizers/{pkg}" if "punkt" in pkg else f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

lemmatizer = WordNetLemmatizer()


def train_chatbot():
    curr_dir = Path(__file__).resolve().parent
    intents_path = curr_dir / "intents.json"
    output_dir = curr_dir.parent.parent / "models" / "chatbot"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(intents_path, "r", encoding="utf-8") as f:
        intents = json.load(f)

    words = []
    classes = []
    documents = []
    ignore_letters = ["?", "!", ".", ","]

    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            w = nltk.word_tokenize(pattern)
            words.extend(w)
            documents.append((w, intent["tag"]))
            if intent["tag"] not in classes:
                classes.append(intent["tag"])

    words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
    words = sorted(list(set(words)))
    classes = sorted(list(set(classes)))

    training = []
    output_empty = [0] * len(classes)

    for doc in documents:
        bag = []
        pattern_words = [lemmatizer.lemmatize(w.lower()) for w in doc[0]]
        for w in words:
            bag.append(1 if w in pattern_words else 0)

        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1
        training.append([bag, output_row])

    random.shuffle(training)
    training = np.array(training, dtype=object)

    train_x = np.array(list(training[:, 0]), dtype=np.float32)
    train_y = np.array(list(training[:, 1]), dtype=np.float32)

    model = Sequential([
        Dense(128, input_shape=(len(train_x[0]),), activation="relu"),
        Dropout(0.5),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(len(train_y[0]), activation="softmax"),
    ])

    sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

    print(f"Training on {len(train_x)} pattern-intent samples across {len(classes)} classes...")
    model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

    model_path = output_dir / "chatbot_model.h5"
    words_path = output_dir / "words.pkl"
    classes_path = output_dir / "classes.pkl"

    model.save(str(model_path))
    with open(words_path, "wb") as f:
        pickle.dump(words, f)
    with open(classes_path, "wb") as f:
        pickle.dump(classes, f)

    print(f"Successfully saved chatbot model to: {model_path}")
    print(f"Saved words to: {words_path}")
    print(f"Saved classes to: {classes_path}")


if __name__ == "__main__":
    train_chatbot()
