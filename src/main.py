from pyannote.audio import Pipeline
import torch
from faster_whisper import WhisperModel
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pydub import AudioSegment
from huggingface_hub import login
import warnings
import logging
import argparse
import os
from collections import defaultdict
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sentence_transformers import CrossEncoder
import nltk
import numpy as np
from tqdm import tqdm
import gdown

from src.plotting import * 
nltk.download('vader_lexicon')
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger('faster_whisper').setLevel(logging.WARNING)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
SAVE_CACHE = True

videos_path = "data/gsocvideos/"
diarization_dir = "cache/diarization_cache/"
cache_dir = "cache/transcription_cache/"
exported_audio_dir = "outputs/exported_audio/"
csv_files = "outputs/csv_files/"
figs_dir = "outputs/figs/"
temp_audio = "tempdata/temp_audio/"

for d in [videos_path, exported_audio_dir, diarization_dir, csv_files, cache_dir, temp_audio, figs_dir]:
    os.makedirs(d, exist_ok=True)

hf_token = "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"
device = "cuda" if torch.cuda.is_available() else "cpu"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"
RED = "\033[91m"

def print_progress(message, color=GREEN):
    print(f"{color}{message}{RESET}")

def load_intent_model():
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')
    return model

def load_sent_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cardiff_model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(cardiff_model_name)
    cardiff_model = AutoModelForSequenceClassification.from_pretrained(cardiff_model_name).to(device)
    return tokenizer, cardiff_model

def load_vader_model():
    return SentimentIntensityAnalyzer()

def load_trans_model():
    model_size = "small.en"
    return WhisperModel(model_size, device=device, compute_type="float16" if device=="cuda" else "int8")

def load_diar_model():
    pipeline_model = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
    pipeline_model.to(torch.device(device))
    return pipeline_model

def transcribe(_model, audio_path, save_cache=SAVE_CACHE):
    cache_file = os.path.join(cache_dir, os.path.basename(audio_path) + ".pkl")
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    segments, _ = _model.transcribe(audio_path, beam_size=5, word_timestamps=False, )
    text = " ".join([seg.text for seg in segments])
    if save_cache:
        with open(cache_file, "wb") as f:
            pickle.dump(text, f)
    return text

def get_diarization(audio_path, diar_model, save_cache=SAVE_CACHE):
    base = os.path.basename(audio_path)
    cache_file = os.path.join(diarization_dir, base + ".pkl")
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    diarization = diar_model(audio_path)
    if save_cache:  
        with open(cache_file, "wb") as f:
            pickle.dump(diarization, f)
    return diarization

intent_labels = [
    "change_speed",
    "lane_change",
    "navigation",
    "traffic_update",
    "weather_update",
    "signal_alert",
    "parking_assistance",
    "fuel_search",
    "vehicle_status",
    "accident_warning",
    "pedestrian_alert",
    "road_block_alert"
]

def analyze_intent(model, text, intent_labels=intent_labels):
    pairs = [(text, intent) for intent in intent_labels]
    scores = model.predict(pairs, show_progress_bar=False)
    confidence_scores = {intent_labels[i]: scores[i] for i in range(len(intent_labels))}
    predicted_intent = max(confidence_scores.items(), key=lambda x: x[1])[0]
    return predicted_intent

def get_vader_sent(analyzer, text):
    scores = analyzer.polarity_scores(text)
    score_array = np.array([scores["neg"], scores["neu"], scores["pos"]])
    label = ["Negative", "Neutral", "Positive"][np.argmax(score_array)]
    return score_array, label

def get_cardiff_sent(tokenizer, cardiff_model, text, max_length=512):
    tokens = tokenizer.encode(text, add_special_tokens=True)
    if len(tokens) > max_length:
        chunked_scores = []
        for i in range(0, len(tokens), max_length):
            chunk = tokenizer.decode(tokens[i : i + max_length])
            inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512).to(device)
            outputs = cardiff_model(**inputs)
            scores = torch.nn.functional.softmax(outputs.logits, dim=-1).detach().cpu().numpy()[0]
            chunked_scores.append(scores)
        
        scores = np.mean(chunked_scores, axis=0)
    else:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length = 512).to(device)
        outputs = cardiff_model(**inputs)
        scores = torch.nn.functional.softmax(outputs.logits, dim=-1).detach().cpu().numpy()[0]
    
    label = ["Negative", "Neutral", "Positive"][np.argmax(scores)]
    return scores, label

def analyze_sentiment(text, tokenizer, cardiff_model, analyzer, w_card=0.6, w_vader=0.4):
    c_scores, _ = get_cardiff_sent(tokenizer, cardiff_model, text)
    v_scores, _ = get_vader_sent(analyzer, text)
    combined_scores = w_card * c_scores + w_vader * v_scores
    label = ["Negative", "Neutral", "Positive"][np.argmax(combined_scores)]
    return label

def chunking_based(segments, tokenizer, cardiff_model, analyzer, intent_model, chunk_time=5, window_size=1):
    rows = []
    current_window_start = 0
    current_window_end = chunk_time
    current_window_text = ""
    window_history = []
    
    segments = list(segments)
    print_progress("[+] Transcription completed", GREEN)

    for segment in tqdm(segments, desc="Processing Segments", bar_format="{l_bar}{bar:30}{r_bar} {n_fmt}/{total_fmt}", colour="cyan"):
        for word in segment.words:
            if word.end <= current_window_end:
                current_window_text += " " + word.word
            else:
                if current_window_text.strip():
                    window_history.append((current_window_text.strip(), current_window_start, current_window_end))
                    
                    start_index = max(0, len(window_history) - window_size)
                    context_window = window_history[start_index:]
                    
                    combined_text = " ".join([text for text, _, _ in context_window])
                    context_start = context_window[0][1]
                    context_end = context_window[-1][2]
                    
                    if not rows or rows[-1]["start"] != context_start or rows[-1]["end"] != context_end:
                        rows.append({
                            "start": context_start,
                            "end": context_end,
                            "text": context_window[-1][0],
                            "sentiment": analyze_sentiment(combined_text, tokenizer, cardiff_model, analyzer),
                            "intent": analyze_intent(intent_model, combined_text)
                        })
                
                while word.end > current_window_end:
                    current_window_start += chunk_time
                    current_window_end += chunk_time
                
                current_window_text = word.word
    
    if current_window_text.strip():
        window_history.append((current_window_text.strip(), current_window_start, current_window_end))
        start_index = max(0, len(window_history) - window_size)
        context_window = window_history[start_index:]
        combined_text = " ".join([text for text, _, _ in context_window])
        context_start = context_window[0][1]
        context_end = context_window[-1][2]
        
        if not rows or rows[-1]["start"] != context_start or rows[-1]["end"] != context_end:
            rows.append({
                "start": context_start,
                "end": context_end,
                "text": context_window[-1][0],
                "sentiment": analyze_sentiment(combined_text, tokenizer, cardiff_model, analyzer),
                "intent": analyze_intent(intent_model, combined_text)
            })
    
    return rows

def sentence_based(segments, tokenizer, cardiff_model, analyzer, intent_model, window_size=1):
    rows = []
    current_sentence = ""
    sentence_start = None
    accumulated_sentences = []
    
    segments = list(segments)
    print_progress("[+] Transcription completed", GREEN)

    for segment in tqdm(segments, desc="Processing Segments", bar_format="{l_bar}{bar:30}{r_bar} {n_fmt}/{total_fmt}", colour="cyan"):
        for word in segment.words:
            if sentence_start is None:
                sentence_start = word.start
            current_sentence += " " + word.word
            
            if word.word.endswith('.'):
                accumulated_sentences.append((current_sentence.strip(), sentence_start, word.end))
                current_sentence = ""
                sentence_start = None
    
    if current_sentence.strip():
        accumulated_sentences.append((current_sentence.strip(), sentence_start, word.end))
    
    for i in range(len(accumulated_sentences)):
        start_index = max(0, i - window_size + 1)
        context_window = accumulated_sentences[start_index:i+1]
        
        combined_text = " ".join([text for text, _, _ in context_window])
        context_start = context_window[0][1]
        context_end = context_window[-1][2]
        
        if not rows or rows[-1]["start"] != context_start or rows[-1]["end"] != context_end:
            rows.append({
                "start": context_start,
                "end": context_end,
                "text": context_window[-1][0],  
                "sentiment": analyze_sentiment(combined_text, tokenizer, cardiff_model, analyzer),
                "intent": analyze_intent(intent_model, combined_text)
            })
    
    return rows

def speaker_wise_text_analysis(speaker_index, speaker_segments, base_name, tokenizer, cardiff_model, analyzer, intent_model, window_size=1):
    rows = []
    contexts = []
    speaker = list(speaker_segments.keys())[speaker_index]
    for conv in tqdm(speaker_segments[speaker], desc="Processing segments for speaker " + speaker):
        contexts.append(conv[2])
        start = max(0, len(contexts) - window_size)
        aggregated_text = " ".join(contexts[start:])
        intent = analyze_intent(intent_model, aggregated_text)
        sentiment = analyze_sentiment(aggregated_text, tokenizer, cardiff_model, analyzer)
        rows.append({
            "start": conv[0],
            "end": conv[1],
            "text": aggregated_text,
            "sentiment": sentiment,
            "intent": intent
        })
    df = pd.DataFrame(rows)
    path = "outputs/csv_files/" + f"{base_name}_SPEAKER_{speaker_index}.csv"
    df.to_csv(path, index=False)
    print_progress(f"[+] saved to {path}", GREEN)

def FUNCTION1_results(trans_model, audio_path, base_name, tokenizer, cardiff_model, analyzer, intent_model, analysis_type="sentence", window_size=1, chunk_time = 5):
    """
    Perform sentiment analysis on transcribed audio.
    analysis_type: "chunk" for time-based analysis, "sentence" for sentence-based.
    window_size: Number of previous segments to accumulate.
    """
    print_progress("[+] Transcribing text", BLUE)
    segments, _ = trans_model.transcribe(audio_path, beam_size=3, word_timestamps=True, log_progress = True)

    if analysis_type == "chunk":
        rows = chunking_based(segments, tokenizer, cardiff_model, analyzer, intent_model, window_size=window_size, chunk_time = chunk_time)
    else:
        rows = sentence_based(segments, tokenizer, cardiff_model, analyzer, intent_model,  window_size=window_size)

    df = pd.DataFrame(rows)
    csv_file_path = os.path.join(csv_files, base_name + ".csv")
    df.to_csv(csv_file_path, index=False)
    return csv_file_path

def compute_speaker_segments(diarization, trans_model, audio_path):
    full_audio = AudioSegment.from_wav(audio_path)
    speakers = sorted({speaker for _, _, speaker in diarization.itertracks(yield_label=True)})
    speaker_segments = defaultdict(list)
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segment_start_ms = int(turn.start * 1000)
        segment_end_ms = int(turn.end * 1000)
        segment_path = os.path.join(temp_audio, f"{speaker}_{turn.start:.1f}_{turn.end:.1f}.wav")
        full_audio[segment_start_ms:segment_end_ms].export(segment_path, format="wav")
        text = transcribe(trans_model, segment_path)
        speaker_segments[speaker].append((turn.start, turn.end, text, "#abcdef"))
    return speaker_segments

def main(videos_folder, trans_model, diar_model, tokenizer, cardiff_model, analyzer, intent_model, analysis_type, window_size, chunk_time):
    for vid in os.listdir(videos_folder):
        video_path = os.path.join(videos_folder, vid)
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        audio_path = os.path.join(exported_audio_dir, f"{base_name}.wav")
        audio = AudioSegment.from_file(video_path)
        audio.export(audio_path, format="wav")
        print_progress(f"[+] processing video: {vid}", color=YELLOW)
        print_progress("[+] processing diarization", BLUE)
        diarization = get_diarization(audio_path, diar_model)
        print_progress("[+] diarization completed", GREEN)
        print_progress("[+] generating csv file with sentiment analysis.", BLUE)
        csv_file_path = FUNCTION1_results(trans_model, audio_path, base_name, tokenizer, cardiff_model, analyzer, intent_model, analysis_type = analysis_type, window_size = window_size, chunk_time = chunk_time)
        print_progress("[+] csv file generated", GREEN)
        print_progress("[+] computing speaker segments.", BLUE)
        speaker_segments = compute_speaker_segments(diarization, trans_model, audio_path)
        print_progress("[+] speaker segmented computed..", GREEN)
        print_progress("[+] plotting figures.", BLUE)
        FUNCTION2_plot_figures(diarization, csv_file_path, len(audio)/1000, trans_model, audio_path, base_name, speaker_segments)
        print_progress("[+] all figures plotted.", GREEN)
        
        for i in range(len(list(speaker_segments.keys()))):
            speaker_wise_text_analysis(i, speaker_segments, base_name, tokenizer, cardiff_model, analyzer, intent_model, window_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process video files for communication analysis.")
    parser.add_argument("folder", nargs="?", default=videos_path, help="Folder containing video files")
    parser.add_argument("--st", type=str, default="chunk", help="Analysis Strategy: 'chunk' or 'sentence'")
    parser.add_argument("--wsz", type=int, default=1, help="Window size for analysis")
    parser.add_argument("--gdrive", type=str, default=None, help="Google Drive folder URL for video retrieval")
    parser.add_argument("--hf_token", type=str, default="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", help="Hugging Face Hub token")
    parser.add_argument("--ct", type=int, default=5, help="Chunk time in seconds for chunk-based analysis")

    args = parser.parse_args()
    print_progress("[+] logging in hf.", YELLOW)
    try:
        login(args.hf_token)
        hf_token = args.hf_token
    except:
        print_progress("change hf token it is expired", RED)

    if args.gdrive:
        print_progress("downloading gdrive videos", YELLOW)
        gdown.download_folder(args.gdrive, output = "data/gsocvideos", quiet=False)

    if(len(os.listdir(videos_path)) == 0):
        print("input data is empty downloading video from drive.")
        folder_url = "https://drive.google.com/drive/folders/1clnoqARUaLfR-fC42w8WkwYmCqfZtoN5"
        gdown.download_folder(folder_url, output = "data/gsocvideos", quiet=False)

    trans_model = load_trans_model()
    print_progress("[+] transcription model loaded.", GREEN)
    diar_model = load_diar_model()
    print_progress("[+] diarization model loaded.", GREEN)
    tokenizer, cardiff_model = load_sent_model()
    analyzer = load_vader_model()
    print_progress("[+] sentiment model loaded.", GREEN)
    intent_model = load_intent_model()
    print_progress("[+] intent model loaded.", GREEN)

    main(args.folder, trans_model, diar_model, tokenizer, cardiff_model, analyzer, intent_model, args.st, args.wsz, args.ct)