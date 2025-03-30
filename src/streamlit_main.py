import streamlit as st
from pydub import AudioSegment
from pyannote.audio import Pipeline
import torch
from faster_whisper import WhisperModel
from transformers import pipeline
from bertopic import BERTopic
import spacy
import os
from huggingface_hub import login
import pandas as pd
import plotly.express as px
from collections import defaultdict
import warnings
import pickle
from streamlit_plotting import *

nlp = spacy.load("en_core_web_sm")
stopwords = nlp.Defaults.stop_words

warnings.filterwarnings("ignore")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

st.set_page_config(layout='centered')
SAVE_CACHE = True

hf_token = "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"
login(hf_token)

videos_path = "data/gsocvideos/"
diarization_dir = "cache/diarization_cache/"
cache_dir = "cache/transcription_cache/"
exported_audio_dir = "outputs/exported_audio/"
csv_files = "outputs/csv_files/"
temp_audio = "tempdata/temp_audio/"
figs_dir = "outputs/figs/"

for d in [videos_path, exported_audio_dir, diarization_dir, csv_files, cache_dir, temp_audio, figs_dir]:
    os.makedirs(d, exist_ok=True)

if len(os.listdir(videos_path)) == 0:
    import gdown
    st.info("Input data empty. Downloading video from drive.")
    folder_url = "https://drive.google.com/drive/folders/1clnoqARUaLfR-fC42w8WkwYmCqfZtoN5"
    gdown.download_folder(folder_url, output = "data/gsocvideos", quiet=False)

device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_intent_model():
    from sentence_transformers import CrossEncoder
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')
    return model

@st.cache_resource
def load_sent_model():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", device=0 if device=="cuda" else -1)

@st.cache_resource
def load_trans_model():
    model_size = "small.en"
    return WhisperModel(model_size, device=device, compute_type="float16" if device=="cuda" else "int8")

@st.cache_resource
def load_diar_model():
    pipeline_model = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
    pipeline_model.to(torch.device(device))
    return pipeline_model

def transcribe(_model, audio_path, save_cache=SAVE_CACHE):
    cache_file = os.path.join(cache_dir, os.path.basename(audio_path) + ".pkl")
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    segments, _ = _model.transcribe(audio_path, beam_size=5, word_timestamps=False)
    text = " ".join([seg.text for seg in segments])
    if save_cache:
        with open(cache_file, "wb") as f:
            pickle.dump(text, f)
    return text

def get_diarization(audio_path, save_cache=SAVE_CACHE):
    base = os.path.basename(audio_path)
    cache_file = os.path.join(diarization_dir, base + ".pkl")
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    diar_model = load_diar_model()
    diarization = diar_model(audio_path)
    if save_cache:
        with open(cache_file, "wb") as f:
            pickle.dump(diarization, f)
    return diarization

def analyze_sentiment(analyzer, text):
    result = analyzer(text)[0]
    sentiment_map = {"negative": "Negative", "neutral": "Neutral", "positive": "Positive"}
    return sentiment_map[result["label"].lower()]

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

def analyze_intent(model, text):
    pairs = [(text, intent) for intent in intent_labels]
    scores = model.predict(pairs)
    confidence_scores = {intent_labels[i]: scores[i] for i in range(len(intent_labels))}
    predicted_intent = max(confidence_scores, key=confidence_scores.get)
    return predicted_intent

def compute_speaker_segments(diarization, trans_model, audio_path):
    full_audio = AudioSegment.from_wav(audio_path)
    speaker_segments = defaultdict(list)
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segment_start_ms = int(turn.start * 1000)
        segment_end_ms = int(turn.end * 1000)
        segment_path = os.path.join(temp_audio, f"{speaker}_{turn.start:.1f}_{turn.end:.1f}.wav")
        full_audio[segment_start_ms:segment_end_ms].export(segment_path, format="wav")
        text = transcribe(trans_model, segment_path)
        speaker_segments[speaker].append((turn.start, turn.end, text))
    return speaker_segments

def clean_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.text not in stopwords and token.is_alpha]
    return ' '.join(tokens)

def FUNCTION1_results(trans_model, sent_model, intent_model, audio_path, base_name):
    rows = []
    current_window_start, current_window_end, current_text = 0, 5, ""
    segments, _ = trans_model.transcribe(audio_path, beam_size=2, word_timestamps=True)
    for segment in segments:
        for word in segment.words:
            if word.end <= current_window_end:
                current_text += " " + word.word
            else:
                if current_text.strip():
                    rows.append({
                        "start": current_window_start,
                        "end": current_window_end,
                        "text": current_text.strip(),
                        "sentiment": analyze_sentiment(sent_model, current_text.strip()),
                        "intent": analyze_intent(intent_model, current_text.strip())
                    })
                while word.end > current_window_end:
                    current_window_start += 5
                    current_window_end += 5
                current_text = word.word
    if current_text.strip():
        rows.append({
            "start": current_window_start,
            "end": current_window_end,
            "text": current_text.strip(),
            "sentiment": analyze_sentiment(sent_model, current_text.strip()),
            "intent": analyze_intent(intent_model, current_text.strip())
        })
    df = pd.DataFrame(rows)
    csv_file_path = os.path.join(csv_files, base_name + ".csv")
    df.to_csv(csv_file_path, index=False)
    return csv_file_path

def FUNCTION2_plot_figures(diarization, csv_file_path, audio_duration, trans_model, audio_path, speaker_segments):
    timeline_fig = plot_speaker_timeline(diarization, audio_duration)
    histogram_fig = generate_histogram(csv_file_path)
    sentiment_fig = generate_sentiment_plot(csv_file_path)
    engagement_fig = plot_engagement_metrics(diarization, audio_duration)
    heatmap_fig = plot_speaker_heatmap(diarization, audio_duration, time_bin_size=5)
    network_fig = plot_interaction_network(diarization)
    total_words_fig = plot_total_word_counts(speaker_segments)
    avg_turn_fig = plot_average_turn_length(speaker_segments)
    cumulative_fig = plot_cumulative_talk_time(speaker_segments, audio_duration, bin_size=10)

    df = pd.read_csv(csv_file_path)
    texts = df['text'].dropna().tolist()
    texts = [clean_text(t) for t in texts]
    topic_model = BERTopic()
    topic_model.fit_transform(texts)

    st.plotly_chart(timeline_fig, use_container_width=True)
    st.plotly_chart(histogram_fig, use_container_width=True)
    st.plotly_chart(sentiment_fig, use_container_width=True)
    st.plotly_chart(engagement_fig, use_container_width=True)
    st.plotly_chart(heatmap_fig, use_container_width=True)
    st.plotly_chart(network_fig, use_container_width=True)
    st.plotly_chart(total_words_fig, use_container_width=True)
    st.plotly_chart(avg_turn_fig, use_container_width=True)
    st.plotly_chart(cumulative_fig, use_container_width=True)
    topic_fig = topic_model.visualize_barchart()
    st.plotly_chart(topic_fig, use_container_width=True)
    return speaker_segments

trans_model = load_trans_model()
diar_model = load_diar_model()
sent_model = load_sent_model()
intent_model = load_intent_model()

st.header(":blue[Communication Analysis] :grey[Dashboard]")

def colored_horizontal_line(color="#FF5733", thickness="3px", margin="10px 0"):
    st.markdown(f"<hr style='border: none; height: {thickness}; background-color: {color}; margin: {margin};'>", unsafe_allow_html=True)

def color_sentiment(val):
    val_lower = str(val).lower()
    if val_lower == 'positive':
        return 'color: green; font-weight: bold;'
    elif val_lower == 'negative':
        return 'color: red; font-weight: bold;'
    elif val_lower == 'neutral':
        return 'color: gray; font-weight: bold;'
    else:
        return 'color: black; font-weight: bold;'

def color_intent(val):
    intent_colors = {
        'change_speed': '#FFA07A',
        'lane_change': '#F08080',
        'navigation': '#87CEFA',
        'traffic_update': '#90EE90',
        'weather_update': '#D8BFD8',
        'signal_alert': '#FFB6C1',
        'parking_assistance': '#FFD700',
        'fuel_search': '#ADFF2F',
        'vehicle_status': '#20B2AA',
        'accident_warning': '#FF4500',
        'pedestrian_alert': '#DAA520',
        'road_block_alert': '#8B0000'
    }
    return f'background-color: {intent_colors.get(val, "#E0E0E0")}; font-weight: bold;'

if st.button("Process Videos"):
    st.info("Starting video processing")
    speaker_segments = ""
    video_files = os.listdir(videos_path)
    bar = st.progress(0, text="Processing Videos")
    for idx, vid in enumerate(video_files):
        bar.progress((idx + 1) / len(video_files))
        video_path = os.path.join(videos_path, vid)
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        with st.spinner(f"Processing {vid}"):
            st.info(f"Extracting audio from {vid}")
            audio_path = os.path.join(exported_audio_dir, f"{base_name}.wav")
            audio = AudioSegment.from_file(video_path)
            audio.export(audio_path, format="wav")
            st.info("Performing diarization")
            diarization = get_diarization(audio_path)
            st.info("Transcribing and generating CSV")
            csv_file_path = FUNCTION1_results(trans_model, sent_model, intent_model, audio_path, base_name)
            st.info("Computing speaker segments")
            speaker_segments = compute_speaker_segments(diarization, trans_model, audio_path)
            st.info("Generating plots")
            FUNCTION2_plot_figures(diarization, csv_file_path, len(audio)/1000, trans_model, audio_path, speaker_segments)
            st.success(f"Completed processing for {vid}")
    colors = px.colors.qualitative.Plotly
    speakers = sorted(speaker_segments.keys())
    color_map = {speaker: colors[i % len(colors)] for i, speaker in enumerate(speakers)}
    cols = st.columns(len(speakers))
    for i, speaker in enumerate(speakers):
        with cols[i]:
            st.subheader(f"Speaker {speaker}")
            st.markdown(f"<hr style='border: none; height: 3px; background-color: {color_map[speaker]}; margin: 10px 0;'>", unsafe_allow_html=True)
            for start, end, text in speaker_segments[speaker]:
                colored_timestamp = f'<span style="color: {color_map[speaker]}; font-weight: bold;">Time: {start:.1f}s - {end:.1f}s</span>'
                st.markdown(colored_timestamp, unsafe_allow_html=True)
                st.markdown(f"<p style='color: black;'>Transcript: {text}</p>", unsafe_allow_html=True)
                audio_file = os.path.join(temp_audio, f"{speaker}_{start:.1f}_{end:.1f}.wav")
                st.audio(audio_file)
            st.markdown(f"<hr style='border: none; height: 3px; background-color: {color_map[speaker]}; margin: 10px 0;'>", unsafe_allow_html=True)
    df = pd.read_csv(csv_file_path)
    df_analysis = df[['text', 'sentiment', 'intent']].dropna()
    styled_table = (
        df_analysis.style
        .applymap(color_sentiment, subset=['sentiment'])
        .applymap(color_intent, subset=['intent'])
        .set_table_styles([{'selector': 'thead th', 'props': [('background-color', '#4CAF50'), ('color', 'white'), ('font-size', '16px'), ('padding', '8px')]}])
        .set_properties(**{'font-size': '14px', 'padding': '8px'})
        .set_caption("Text Analysis")
    )
    st.markdown("<h2 style='text-align: center; color:#4CAF50;'>Text Analysis</h2>", unsafe_allow_html=True)
    st.write(styled_table.to_html(), unsafe_allow_html=True)
