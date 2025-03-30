import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.main import transcribe, get_diarization, analyze_sentiment, load_sent_model, load_trans_model, load_diar_model
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
    model = load_sent_model()
    positive_text = "This is a good example."
    negative_text = "This is a bad example."
    neutral_text = "This is an example."
    assert analyze_sentiment(model, positive_text) == "Positive"
    assert analyze_sentiment(model, negative_text) == "Negative"
    assert analyze_sentiment(model, neutral_text) == "Neutral"

def test_get_diarization():
    test_diar_audio_path = "tests/test_data/test_diar_audio.wav"
    diar = get_diarization(test_diar_audio_path)
    diar_pickle_path = "tests/test_data/diar.pkl"
    with open(diar_pickle_path, "rb") as f:
        real_diarization = pickle.load(f)["diarization"]
    assert len(list(diar.itertracks(yield_label=True))) == len(list(real_diarization.itertracks(yield_label=True)))