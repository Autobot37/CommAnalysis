import streamlit as st
import os, pickle, warnings
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from pydub import AudioSegment
from pyannote.audio import Pipeline
import torch
from faster_whisper import WhisperModel
from collections import defaultdict
from flair.nn import Classifier
from flair.data import Sentence
from sentence_transformers import CrossEncoder
from bertopic import BERTopic
import spacy
nlp = spacy.load("en_core_web_sm")
stopwords = nlp.Defaults.stop_words

warnings.filterwarnings("ignore")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

st.set_page_config(layout = 'centered')
# Directories and settings
videos_path_default = "vids/"
exported_audio_dir = "exported_audio"
diarization_dir = "diarization_cache"
cache_dir = "transcription_cache"
csv_files = "csv_files"
temp_audio = "temp_audio"
for d in [exported_audio_dir, diarization_dir, csv_files, cache_dir, temp_audio]:
    os.makedirs(d, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
sentiment_colors = {"Positive": "#4CAF50", "Negative": "#F44336", "Neutral": "#9E9E9E"}

@st.cache_resource
def load_sent_model():
    return Classifier.load('sentiment')

@st.cache_resource
def load_intent_model():
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')
    return model

@st.cache_resource
def load_trans_model():
    model_size = "tiny.en"
    return WhisperModel(model_size, device=device, compute_type="float16" if device=="cuda" else "int8")

@st.cache_resource
def load_diar_model():
    pipeline_model = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", 
                                               use_auth_token="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")
    pipeline_model.to(torch.device(device))
    return pipeline_model

def transcribe(_model, audio_path):
    cache_file = os.path.join(cache_dir, os.path.basename(audio_path) + ".pkl")
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    segments, _ = _model.transcribe(audio_path, beam_size=2, word_timestamps=False)
    text = " ".join([seg.text for seg in segments])
    with open(cache_file, "wb") as f:
        pickle.dump(text, f)
    return text

def get_diarization(audio_path):
    base = os.path.basename(audio_path)
    cache_file = os.path.join(diarization_dir, base + ".pkl")
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    diar_model = load_diar_model()
    diarization = diar_model(audio_path)
    with open(cache_file, "wb") as f:
        pickle.dump(diarization, f)
    return diarization

def analyze_sentiment(analyzer, text):
    sentence = Sentence(text)
    analyzer.predict(sentence)
    tag = sentence.tag
    score = sentence.score
    if score < 0.85:
        return "Neutral"
    return {"NEGATIVE": "Negative", "POSITIVE": "Positive"}[tag]

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

def clean_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.text not in stopwords and token.is_alpha]
    return ' '.join(tokens)

def plot_speaker_timeline(diarization, total_duration):
    speakers = sorted({speaker for _, _, speaker in diarization.itertracks(yield_label=True)})
    colors = px.colors.qualitative.Plotly
    color_map = {speaker: colors[i % len(colors)] for i, speaker in enumerate(speakers)}
    
    fig = go.Figure()
    for speaker in speakers:
        xs, ys = [], []
        for turn, _, spk in diarization.itertracks(yield_label=True):
            if spk == speaker:
                xs.extend([turn.start, turn.end, None])
                ys.extend([f"Speaker {speaker}", f"Speaker {speaker}", None])
        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(color=color_map[speaker], width=10),
            name=f"Speaker {speaker}"
        ))
    fig.update_layout(
        title="Speaker Activity Timeline",
        xaxis_title="Time (seconds)",
        yaxis_title="Speaker",
        xaxis=dict(range=[0, total_duration], type="linear"),
        yaxis=dict(autorange="reversed"),
        height=300 + 50 * len(speakers)
    )
    return fig

def generate_histogram(csv_file):
    df = pd.read_csv(csv_file)
    df['bucket_order'] = df['start'] // 5
    df = df.sort_values('bucket_order')
    def make_label(start):
        start_sec = int(start)
        end_sec = start_sec + 5
        return f"{start_sec//60:02d}:{start_sec%60:02d}-{end_sec//60:02d}:{end_sec%60:02d}"
    
    df['bucket_label'] = df['start'].apply(make_label)
    df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
    
    fig = px.bar(
        df, 
        x="bucket_label", 
        y="word_count", 
        color="sentiment",
        category_orders={"bucket_label": df['bucket_label'].unique()}, 
        color_discrete_map=sentiment_colors,
        labels={"bucket_label": "Time Bucket", "word_count": "Word Count"}
    )
    
    fig.update_layout(
        title="Word Count per 5-Second Segment",
        xaxis_tickangle=-90,
        xaxis={'type': 'category', 'categoryorder': 'array', 'categoryarray': df['bucket_label'].unique()}
    )
    return fig

def generate_sentiment_plot(csv_file):
    df = pd.read_csv(csv_file)
    counts = df['sentiment'].value_counts().reindex(["Positive", "Negative", "Neutral"], fill_value=0).reset_index()
    counts.columns = ["Sentiment", "Count"]
    fig = px.bar(counts, x="Sentiment", y="Count", color="Sentiment", color_discrete_map=sentiment_colors,
                 title="Sentiment Distribution")
    return fig

def plot_engagement_metrics(diarization, total_duration):
    speaker_stats = defaultdict(lambda: {'talk_time': 0, 'turns': 0})
    for track, _, speaker in diarization.itertracks(yield_label=True):
        speaker_stats[speaker]['talk_time'] += track.end - track.start
        speaker_stats[speaker]['turns'] += 1
    speakers = [f"Speaker {s}" for s in speaker_stats.keys()]
    stats = pd.DataFrame({
        "Speaker": speakers,
        "Talk Time": [v['talk_time'] for v in speaker_stats.values()],
        "Turn Count": [v['turns'] for v in speaker_stats.values()]
    })
    fig = px.bar(stats, x="Speaker", y=["Talk Time", "Turn Count"], barmode="group", title="Engagement Metrics")
    return fig

def plot_speaker_heatmap(diarization, total_duration, time_bin_size=5):
    bins = np.arange(0, int(total_duration) + time_bin_size, time_bin_size)
    speakers = sorted({speaker for _, _, speaker in diarization.itertracks(yield_label=True)})
    print(len(speakers))
    activity_matrix = np.zeros((len(speakers), len(bins) - 1))
    for track, _, speaker in diarization.itertracks(yield_label=True):
        i = speakers.index(speaker)
        bin_idx_start = np.digitize(track.start, bins) - 1
        bin_idx_end = np.digitize(track.end, bins) - 1
        activity_matrix[i, bin_idx_start:bin_idx_end + 1] += 1
    df = pd.DataFrame(activity_matrix, index=[f"Speaker {s}" for s in speakers], columns=[f"{int(b)}s" for b in bins[:-1]])
    fig = px.imshow(df, labels=dict(x="Time Bin", y="Speaker", color="Activity"),
                    x=df.columns, y=df.index, title="Speaker Activity Heatmap")
    return fig

def plot_interaction_network(diarization):
    interactions = defaultdict(int)
    prev_speaker = None
    for _, _, current_speaker in diarization.itertracks(yield_label=True):
        if prev_speaker is not None and prev_speaker != current_speaker:
            interactions[(prev_speaker, current_speaker)] += 1
        prev_speaker = current_speaker

    G = nx.DiGraph()
    for (s1, s2), weight in interactions.items():
        G.add_edge(f"Speaker {s1}", f"Speaker {s2}", weight=weight)

    pos = nx.spring_layout(G, seed=42)
    
    edge_x = []
    edge_y = []
    edge_labels_x = []
    edge_labels_y = []
    edge_texts = []
    for (u, v, data) in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        # Compute midpoint for label
        edge_labels_x.append((x0 + x1) / 2)
        edge_labels_y.append((y0 + y1) / 2)
        edge_texts.append(str(data['weight']))

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    edge_label_trace = go.Scatter(
        x=edge_labels_x,
        y=edge_labels_y,
        text=edge_texts,
        mode='text',
        textfont=dict(color='black', size=12),
        hoverinfo='none'
    )

    node_x = []
    node_y = []
    node_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="bottom center",
        marker=dict(size=20, color='#ffc107'),
        hoverinfo='text'
    )

    fig = go.Figure(data=[edge_trace, edge_label_trace, node_trace],
                    layout=go.Layout(
                        title="Speaker Interaction Network",
                        showlegend=False,
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))
    return fig

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
        speaker_segments[speaker].append((turn.start, turn.end, text))
    return speaker_segments

def plot_total_word_counts(speaker_segments):
    speakers, total_words = [], []
    for speaker, segments in speaker_segments.items():
        count = sum(len(text.split()) for _, _, text in segments)
        speakers.append(f"Speaker {speaker}")
        total_words.append(count)
    fig = px.bar(x=speakers, y=total_words, labels={"x": "Speaker", "y": "Total Word Count"},
                 title="Total Word Count per Speaker", color=speakers)
    return fig

def plot_average_turn_length(speaker_segments):
    speakers, avg_turn = [], []
    for speaker, segments in speaker_segments.items():
        lengths = [end - start for start, end, text in segments]
        avg = sum(lengths) / len(lengths) if lengths else 0
        speakers.append(f"Speaker {speaker}")
        avg_turn.append(avg)
    fig = px.bar(x=speakers, y=avg_turn, labels={"x": "Speaker", "y": "Avg Turn Length (s)"},
                 title="Average Turn Length per Speaker", color=speakers)
    return fig

def plot_cumulative_talk_time(speaker_segments, total_duration, bin_size=10):
    speakers = sorted(speaker_segments.keys())
    bins = np.arange(0, total_duration + bin_size, bin_size)
    cumulative_data = {speaker: np.zeros(len(bins)-1) for speaker in speakers}
    for speaker, segments in speaker_segments.items():
        for start, end, text in segments:
            bin_idx_start = np.digitize(start, bins) - 1
            bin_idx_end = np.digitize(end, bins) - 1
            for i in range(bin_idx_start, bin_idx_end+1):
                cumulative_data[speaker][i] += min(end, bins[i+1]) - max(start, bins[i])
    fig = go.Figure()
    for speaker in speakers:
        fig.add_trace(go.Scatter(x=bins[:-1], y=np.cumsum(cumulative_data[speaker]),
                                 mode='lines+markers', name=f"Speaker {speaker}"))
    fig.update_layout(title="Cumulative Talk Time per Speaker", xaxis_title="Time (s)", yaxis_title="Cumulative Talk Time (s)")
    return fig

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
                        "intent" : analyze_intent(intent_model, current_text.strip())
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
            "intent" : analyze_intent(intent_model, current_text.strip())
        })
    df = pd.DataFrame(rows)
    csv_file_path = os.path.join(csv_files, base_name + ".csv")
    df.to_csv(csv_file_path, index=False)
    return csv_file_path

def FUNCTION2_plot_figures(diarization, csv_file_path, audio_duration, trans_model, audio_path, base_name):
    timeline_fig = plot_speaker_timeline(diarization, audio_duration)
    histogram_fig = generate_histogram(csv_file_path)
    sentiment_fig = generate_sentiment_plot(csv_file_path)
    engagement_fig = plot_engagement_metrics(diarization, audio_duration)
    heatmap_fig = plot_speaker_heatmap(diarization, audio_duration, time_bin_size=5)
    network_fig = plot_interaction_network(diarization)
    speaker_segments = compute_speaker_segments(diarization, trans_model, audio_path)
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
videos_path = "vids"
def colored_horizontal_line(color="#FF5733", thickness="3px", margin="10px 0"):
    st.markdown(
        f"<hr style='border: none; height: {thickness}; background-color: {color}; margin: {margin};'>",
        unsafe_allow_html=True
    )

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
    speaker_segments = ""
    video_files = os.listdir(videos_path)
    bar = st.progress(0, text="Processing Videos")
    for idx, vid in enumerate(video_files):
        bar.progress((idx + 1)/len(video_files))
        video_path = os.path.join(videos_path, vid)
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        with st.spinner(f"Processing {vid}..."):
            audio_path = os.path.join(exported_audio_dir, f"{base_name}.wav")
            audio = AudioSegment.from_file(video_path)
            audio.export(audio_path, format="wav")
            diarization = get_diarization(audio_path)
            csv_file_path = FUNCTION1_results(trans_model, sent_model, intent_model, audio_path, base_name)
            print("csv file created.")
            speaker_segments = FUNCTION2_plot_figures(diarization, csv_file_path, len(audio)/1000, trans_model, audio_path, base_name)
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
        .set_table_styles([
            {'selector': 'thead th',
            'props': [('background-color', '#4CAF50'),
                    ('color', 'white'),
                    ('font-size', '16px'),
                    ('padding', '8px')]}
        ])
        .set_properties(**{'font-size': '14px', 'padding': '8px'})
        .set_caption("Text Analysis")
    )

    st.markdown("<h2 style='text-align: center; color:#4CAF50;'>Text Analysis</h2>", unsafe_allow_html=True)
    st.write(styled_table.to_html(), unsafe_allow_html=True)