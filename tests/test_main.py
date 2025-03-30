import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.main import transcribe, get_diarization, analyze_sentiment, load_sent_model, load_trans_model, load_diar_model
import pickle

def test_transcribe():
    print("testing transcription")
    audio_paths = ["tests/test_data/test_audio_0.wav", "tests/test_data/test_audio_1.wav"]
    with open("tests/test_data/test_audio_text.txt", "r") as f:
        real_texts = f.read().splitlines()
    model = load_trans_model()
    texts = [transcribe(model, p) for p in audio_paths]
    real_texts = [t.strip() for t in real_texts]
    texts = [t.strip() for t in texts]
    assert texts == real_texts, "Texts do not match!"

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