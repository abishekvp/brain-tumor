from django.shortcuts import render
from django.http import JsonResponse
# import random
# import json
import numpy as np
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
#from .models import Response, models
from .processor import chatbot_response
# Remove the comments to download additional nltk packages
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST


# import nltk
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

# from sklearn.feature_extraction.text import TfidfVectorizer

# from tensorflow import keras
# import tensorflow as tf 
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense, Dropout

def index(request):
    return render(request, "app/chatbot.html")


def _translate_text(text: str, target_lang: str) -> str:
    if not text or target_lang == "en":
        return text
    try:
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source="auto", target=target_lang).translate(text)
    except Exception:
        pass
    try:
        from googletrans import Translator
        translator = Translator()
        res = translator.translate(text, dest=target_lang)
        if hasattr(res, "text"):
            return res.text
    except Exception:
        pass
    return text


# Language mapping (browser code → translator language code)
LANG_MAP = {
    'en-US': 'en',
    'hi-IN': 'hi',
    'ta-IN': 'ta',
    'te-IN': 'te',
    'kn-IN': 'kn',
    'ml-IN': 'ml',
}


@csrf_exempt
@require_POST
def chatbot_response_view(request):
    question = request.POST.get('question', '').strip()
    language = request.POST.get('language', 'en-US')   # coming from frontend

    if not question:
        return JsonResponse({"response": "Please type a question."})

    target_lang = LANG_MAP.get(language, 'en')

    try:
        # 1. Translate user question → English (if needed)
        if target_lang != 'en':
            question_en = _translate_text(question, 'en')
        else:
            question_en = question

        # 2. Get answer from existing AI logic
        answer_en = chatbot_response(question_en)

        # 3. Translate answer back to selected language
        if target_lang != 'en':
            answer = _translate_text(answer_en, target_lang)
        else:
            answer = answer_en

        return JsonResponse({"response": answer})

    except Exception as e:
        print("Chatbot error:", e)
        return JsonResponse({
            "response": "Sorry, I encountered an error. Please try again."
        })