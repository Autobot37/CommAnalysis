import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.main import load_vader_model, transcribe, get_diarization, analyze_sentiment, load_sent_model, load_trans_model, load_diar_model, analyze_intent, load_intent_model
import pickle
import re

def test_transcribe():
    audio_path = "tests/test_data/test_audio.mp3"
    with open("tests/test_data/test_audio_text.txt", "r") as f:
        real_text = f.read()
    model = load_trans_model()
    text = transcribe(model, audio_path)
    def clean_text(text: str) -> str:
        return re.sub(r'[^A-Za-z0-9]', '', text)
    text = clean_text(text)
    real_text = clean_text(real_text)
    assert text == real_text

def test_analyze_sentiment():
    print("sentiment")
    tokenizer, cardiff_model = load_sent_model()
    vader = load_vader_model()
    positive_text = "This is a good example."
    negative_text = "This is a bad example."
    neutral_text = "This is an example."
    assert analyze_sentiment(positive_text,tokenizer,cardiff_model, vader) == "Positive"
    assert analyze_sentiment(negative_text,tokenizer,cardiff_model, vader) == "Negative"
    assert analyze_sentiment(neutral_text,tokenizer,cardiff_model, vader) == "Neutral"

def test_get_diarization():
    test_diar_audio_path = "tests/test_data/test_diar_audio.wav"
    model = load_diar_model()
    diar = get_diarization(test_diar_audio_path, model)
    diar_pickle_path = "tests/test_data/diar.pkl"
    with open(diar_pickle_path, "rb") as f:
        real_diarization = pickle.load(f)["diarization"]
    assert len(list(diar.itertracks(yield_label=True))) == len(list(real_diarization.itertracks(yield_label=True)))

def test_intent():
    intent_labels = ["accident", "navigation"]
    model = load_intent_model()
    test_texts = [
        "There is a collision on the highway with multiple vehicles involved.",
        "Go to left from here."
    ]
    pred = [analyze_intent(model, t, intent_labels) for t in test_texts]
    assert pred == intent_labels