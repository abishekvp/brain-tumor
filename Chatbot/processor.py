import json
import os
import pickle
import random
import re
from pathlib import Path

import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

for _pkg in ("punkt", "punkt_tab", "wordnet", "omw-1.4"):
    try:
        nltk.data.find(f"tokenizers/{_pkg}" if "punkt" in _pkg else f"corpora/{_pkg}")
    except LookupError:
        try:
            nltk.download(_pkg, quiet=True)
        except Exception:
            pass

lemmatizer = WordNetLemmatizer()

try:
    from gtts import gTTS
except ImportError:
    gTTS = None

try:
    from pygame import mixer
except ImportError:
    mixer = None

_CHATBOT_DIR = Path(__file__).resolve().parent
_MODELS_CHATBOT_DIR = _CHATBOT_DIR.parent.parent / "models" / "chatbot"

model = load_model(str(_MODELS_CHATBOT_DIR / "chatbot_model.h5"), compile=False)

# Load intents, words, and classes
with open(_CHATBOT_DIR / "intents.json", encoding="utf-8") as _f:
    intents = json.load(_f)
with open(_MODELS_CHATBOT_DIR / "words.pkl", "rb") as _f:
    words = pickle.load(_f)
with open(_MODELS_CHATBOT_DIR / "classes.pkl", "rb") as _f:
    classes = pickle.load(_f)



# -----------------------------
# Helper Functions
# -----------------------------

def safe_init_mixer():
    """Initialize pygame mixer safely."""
    if mixer is None:
        return False
    try:
        if not mixer.get_init():
            mixer.init()
        return True
    except Exception as e:
        print(f"[WARNING] Audio init skipped: {e}")
        return False


def clean_up_sentence(sentence, source_lang='auto', target_lang='en'):
    try:
        sentence_words = nltk.word_tokenize(sentence)
    except Exception:
        sentence_words = re.findall(r"\b\w+\b", sentence)
    try:
        sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    except Exception:
        sentence_words = [word.lower() for word in sentence_words]
    return sentence_words


def bow(sentence, words, show_details=False):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag:", w)
    return np.array(bag)


def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]), verbose=0)[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json, source_lang='en', target_lang='en',
                save_path=None):
    if save_path is None:
        save_path = str(_CHATBOT_DIR / "response.mp3")

    if not ints:
        return "Sorry, I didn't understand that."

    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']

    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])

            # Optional local TTS audio playback if gTTS and pygame are available
            if gTTS is not None and mixer is not None:
                try:
                    tts = gTTS(text=result, lang='en')
                    tts.save(save_path)
                    if safe_init_mixer():
                        sound = mixer.Sound(save_path)
                        sound.play()
                except Exception as e:
                    print(f"[WARNING] Audio playback skipped: {e}")

            return result

    return "You must ask the right questions."


def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res