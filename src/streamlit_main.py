from pyannote.audio import Pipeline
import torch
from faster_whisper import WhisperModel
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pydub import AudioSegment
from huggingface_hub import login
import warnings
import os
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sentence_transformers import CrossEncoder
import nltk
from tqdm import tqdm
import streamlit as st
import json
import base64
from streamlit.components.v1 import html
from src.streamlit_plotting import *
from src.main import get_diarization, FUNCTION1_results, compute_speaker_segments, speaker_wise_text_analysis

st.set_page_config(layout="wide")

st.header(":blue[Communication Analysis] :grey[Dashboard]")

nltk.download('vader_lexicon')
warnings.filterwarnings("ignore")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

SAVE_CACHE = True

hf_token = "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"
if "login" not in st.session_state or not st.session_state.login:
    login(hf_token)
    st.session_state.login = True

videos_path = "data/gsocvideos/"
diarization_dir = "cache/diarization_cache/"
cache_dir = "cache/transcription_cache/"
exported_audio_dir = "outputs/exported_audio/"
csv_files = "outputs/csv_files/"
figs_dir = "outputs/figs/"
temp_audio = "tempdata/temp_audio/"

for d in [videos_path, exported_audio_dir, diarization_dir, csv_files, cache_dir, temp_audio, figs_dir]:
    os.makedirs(d, exist_ok=True)

if(len(os.listdir(videos_path)) == 0):
    import gdown
    print("input data is empty downloading video from drive.")
    folder_url = "https://drive.google.com/drive/folders/1clnoqARUaLfR-fC42w8WkwYmCqfZtoN5"
    gdown.download_folder(folder_url, output = "data/gsocvideos", quiet=False)

uploaded_files = st.file_uploader("Upload Video Files (Multiple)", type=["mp4", "avi", "mov"], accept_multiple_files=True)

if uploaded_files:
    progress_bar = st.progress(0)
    for i, file in enumerate(uploaded_files):
        file_path = os.path.join("data/gsocvideos", file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        progress_bar.progress((i + 1) / len(uploaded_files))
    st.success("All files uploaded successfully!")

device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_intent_model():
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')
    return model

@st.cache_resource
def load_sent_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cardiff_model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(cardiff_model_name)
    cardiff_model = AutoModelForSequenceClassification.from_pretrained(cardiff_model_name).to(device)
    return tokenizer, cardiff_model

@st.cache_resource
def load_vader_model():
    return SentimentIntensityAnalyzer()

@st.cache_resource
def load_trans_model():
    model_size = "small.en"
    return WhisperModel(model_size, device=device, compute_type="float16" if device=="cuda" else "int8")

@st.cache_resource
def load_diar_model():
    pipeline_model = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
    pipeline_model.to(torch.device(device))
    return pipeline_model

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

    st.plotly_chart(timeline_fig, use_container_width=True)
    st.plotly_chart(histogram_fig, use_container_width=True)
    st.plotly_chart(sentiment_fig, use_container_width=True)
    st.plotly_chart(engagement_fig, use_container_width=True)
    st.plotly_chart(heatmap_fig, use_container_width=True)
    st.plotly_chart(network_fig, use_container_width=True)
    st.plotly_chart(total_words_fig, use_container_width=True)
    st.plotly_chart(avg_turn_fig, use_container_width=True)
    st.plotly_chart(cumulative_fig, use_container_width=True)
    return speaker_segments

def colored_horizontal_line(color="#FF5733", thickness="3px", margin="10px 0"):
    st.markdown(f"<hr style='border: none; height: {thickness}; background-color: {color}; margin: {margin};'>", unsafe_allow_html=True)

def color_sentiment(val):
    v = str(val).lower()
    if v == 'positive': return 'color: green; font-weight: bold;'
    if v == 'negative': return 'color: red; font-weight: bold;'
    if v == 'neutral': return 'color: gray; font-weight: bold;'
    return 'color: black; font-weight: bold;'

def color_intent(val):
    d = {'change_speed':'#FFA07A','lane_change':'#F08080','navigation':'#87CEFA','traffic_update':'#90EE90',
         'weather_update':'#D8BFD8','signal_alert':'#FFB6C1','parking_assistance':'#FFD700','fuel_search':'#ADFF2F',
         'vehicle_status':'#20B2AA','accident_warning':'#FF4500','pedestrian_alert':'#DAA520','road_block_alert':'#8B0000'}
    return f'background-color: {d.get(val, "#E0E0E0")}; font-weight: bold;'

p = st.progress(0)
s = st.empty()
s.info("Loading transcription model...")
trans_model = load_trans_model(); p.progress(20)
s.info("Loading diarization model...")
diar_model = load_diar_model(); p.progress(40)
s.info("Loading sentiment model...")
tokenizer, sent_model = load_sent_model(); p.progress(60)
s.info("Loading intent model...")
intent_model = load_intent_model(); p.progress(80)
s.info("Loading VADER model...")
analyzer = load_vader_model(); p.progress(100)
s.success("All models loaded!")

with st.sidebar:
    analysis_type = st.selectbox("Strategy", ["chunk", "sentence"])
    chunk_duration = st.selectbox("Chunk Duration", [5, 10, 15, 20]) if analysis_type == "chunk" else None
    window_size = st.selectbox("Window Size", [1, 3, 5, 7])
    process_videos = st.button("Process Videos")

if process_videos:
    st.info("Starting video processing. This may take several minutes...")
    st.session_state.all_csv_file_path = []
    st.session_state.all_speaker_segments = []
    st.session_state.all_diarizations = []
    st.session_state.all_len_audio = []
    st.session_state.speakers = []
    video_files = os.listdir(videos_path)
    progress_bar = st.progress(0)
    for idx, vid in enumerate(video_files):
        progress_bar.progress((idx + 1) / len(video_files))
        barv = st.progress(0)
        st.info(f"Processing video: {vid}")

        barv.progress(1/5, "exporting audio")
        video_path = os.path.join(videos_path, vid)
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        audio_path = os.path.join(exported_audio_dir, f"{base_name}.wav")
        audio_path_mp3 = os.path.join(exported_audio_dir, f"{base_name}.mp3")
        audio = AudioSegment.from_file(video_path)
        audio.export(audio_path, format="wav")
        audio.export(audio_path_mp3, format="mp3")
        barv.progress(2/5, "diarizing audio")
        diarization = get_diarization(audio_path, diar_model)
        barv.progress(3/5, "generating csv with text analysis")
        csv_file_path = FUNCTION1_results(trans_model, audio_path, base_name, tokenizer, sent_model, analyzer, intent_model, analysis_type, window_size, chunk_duration if chunk_duration else 5)
        barv.progress(4/5, "computing speaker segments")
        speaker_segments = compute_speaker_segments(diarization, trans_model, audio_path)
        st.session_state.all_diarizations.append(diarization)
        st.session_state.all_csv_file_path.append(csv_file_path)
        st.session_state.all_speaker_segments.append(speaker_segments)
        st.session_state.all_len_audio.append(len(audio)/1000)
        st.session_state.speakers.append(list(speaker_segments.keys()))
        barv.progress(5/5, "done")

    st.success("Video processing completed.")

with st.sidebar:
    show_plots = st.button("Show Plots")
    show_speaker_segments = st.button("Show Speakers Segments")
    show_csv = st.button("Show CSV")
    live_audio = st.button("Live Audio Analysis")

if "all_csv_file_path" in st.session_state and st.session_state.all_csv_file_path:
    st.info("Videos are processed already.")
    selected_csv = st.selectbox("Select Video Statistics", st.session_state.all_csv_file_path)
    st.session_state.selected_index = st.session_state.all_csv_file_path.index(selected_csv)
    idx = st.session_state.selected_index
    speaker_segments = st.session_state.all_speaker_segments[idx]
    diarization = st.session_state.all_diarizations[idx]
    csv_file_path = st.session_state.all_csv_file_path[idx]
    base_name = os.path.splitext(os.path.basename(csv_file_path))[0]
    audio_path = os.path.join(exported_audio_dir, f"{base_name}.wav")
    len_audio = st.session_state.all_len_audio[idx]
    speakers = st.session_state.speakers[0] + ["all"]
else:
    st.info("Please process videos.")

if "speakers" in st.session_state:
    with st.sidebar:
        speaker_choice = st.selectbox("Choose speaker for speaker segments.", speakers)

if show_plots:
    st.info("Generating plots...")
    FUNCTION2_plot_figures(diarization, csv_file_path, len_audio, trans_model, audio_path, speaker_segments)

if show_speaker_segments:
    import plotly.express as px
    colors = px.colors.qualitative.Plotly
    index = speakers.index(speaker_choice)
    if index != len(speakers) - 1:
        speakers = [speaker_choice]
    else:
        speakers = speakers[:-1]
    st.info(speaker_segments.keys())
    color_map = {s: colors[i % len(colors)] for i, s in enumerate(speakers)}
    cols = st.columns(len(speakers))
    for i, s in enumerate(speakers):
        with cols[i]:
            st.subheader(f"Speaker {s}")
            st.markdown(f"<hr style='border: none; height: 3px; background-color: {color_map[s]}; margin: 10px 0;'>", unsafe_allow_html=True)
            for start, end, text, _ in speaker_segments[s]:
                st.markdown(f'<span style="color: {color_map[s]}; font-weight: bold;">Time: {start:.1f}s - {end:.1f}s</span>', unsafe_allow_html=True)
                st.markdown(f"<p style='color: black;'>Transcript: {text}</p>", unsafe_allow_html=True)
                af = os.path.join(temp_audio, f"{s}_{start:.1f}_{end:.1f}.wav")
                st.audio(af)
            st.markdown(f"<hr style='border: none; height: 3px; background-color: {color_map[s]}; margin: 10px 0;'>", unsafe_allow_html=True)

if show_csv:
    st.info("Loading CSV data...")
    df = pd.read_csv(csv_file_path)
    df_analysis = df[['text', 'sentiment', 'intent']]
    styled_table = (df_analysis.style
                    .applymap(color_sentiment, subset=['sentiment'])
                    .applymap(color_intent, subset=['intent'])
                    .set_table_styles([{'selector': 'thead th', 'props': [('background-color', '#4CAF50'),
                                                                          ('color', 'white'),
                                                                          ('font-size', '16px'),
                                                                          ('padding', '8px')]}])
                    .set_properties(**{'font-size': '14px', 'padding': '8px'})
                    .set_caption("Text Analysis"))
    st.write("Text Analysis")
    st.write(styled_table.to_html(), unsafe_allow_html=True)

if live_audio:
    st.info("Starting live audio analysis...")
    bn = os.path.basename(csv_file_path).split('.')[0]
    csv_files = []
    for i in range(len(list(speaker_segments.keys()))):
        csv_path = "outputs/csv_files/" + f"{bn}_SPEAKER_{i}.csv"
        csv_files.append(csv_path)
    if not os.path.exists(csv_files[0]):
        st.info("Computing speaker segmentstext analysis")
        for i in range(len(list(speaker_segments.keys()))):
            speaker_wise_text_analysis(i, speaker_segments, base_name, tokenizer, sent_model, analyzer, intent_model, window_size)

    audio_path = os.path.join(exported_audio_dir, f"{bn}.mp3")
    with open("speaker_segments.pkl", "rb") as f:
        speaker_segments = pickle.load(f)
    audio_with_synced_animated_transcript_columns(audio_path, speaker_segments, bn)