### run as python script.py /path/to/videos

import os
import pandas as pd
from pydub import AudioSegment
from pyannote.audio import Pipeline
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from faster_whisper import WhisperModel
from collections import defaultdict
import pickle
from transformers import pipeline
import seaborn as sns
import networkx as nx
import warnings
import numpy as np
from flair.nn import Classifier
from flair.data import Sentence
import argparse

warnings.filterwarnings("ignore")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Directories
videos_path = "vids/"#only parameter to change.
exported_audio_dir = "exported_audio"
diarization_dir = "diarization_cache"
cache_dir="transcription_cache"
csv_files = "csv_files"
temp_audio = "temp_audio"
figs_dir = "figs"
# Create needed directories
for directory in [exported_audio_dir, diarization_dir, csv_files, cache_dir, temp_audio, figs_dir]:
    os.makedirs(directory, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Set a pleasing theme and color palettes
sns.set_theme(style="whitegrid")
sentiment_colors = {"Positive": "#4CAF50", "Negative": "#F44336", "Neutral": "#9E9E9E"}
cmap = LinearSegmentedColormap.from_list("custom_heatmap", ["#f7fbff", "#08306b"])
palette = sns.color_palette("husl", 8)

def load_sent_model():
    """
    Load a multilingual sentiment model for sentiment analysis.
    """
    tagger = Classifier.load('sentiment')
    return tagger

def load_trans_model():
    """
    Load a Whisper model for transcription.
    """
    model_size = "tiny.en"
    return WhisperModel(model_size, device=device, compute_type="float16" if device == "cuda" else "int8")

def load_diar_model():
    """
    Load a speaker diarization model.
    : it segments audio by embedding it and then diarize it in 10sec chunks.
    """
    pipeline_model = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token="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"
    )
    pipeline_model.to(torch.device(device))
    return pipeline_model

def transcribe(_model, audio_path):
    """
    Transcribe audio, caching results to avoid re-computation.
    """
    cache_file = os.path.join(cache_dir, os.path.basename(audio_path) + ".pkl")
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    segments, info = _model.transcribe(audio_path, beam_size=2, word_timestamps=False)
    text = " ".join([seg.text for seg in segments])
    with open(cache_file, "wb") as f:
        pickle.dump(text, f)
    return text

def get_diarization(audio_path):
    """
    Get speaker diarization from a cached file or run the model if not cached.
    """
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
    """
    Run the sentiment model and map output labels to a simple Positive/Negative/Neutral format.
    """
    sentence = Sentence(text)
    analyzer.predict(sentence)
    tag = sentence.tag
    score = sentence.score
    if score < 0.85:
        return "Neutral"
    sentiment_map = {"NEGATIVE": "Negative", "POSITIVE": "Positive"}
    return sentiment_map[tag]

def plot_speaker_timeline(diarization, total_duration):
    """
    Plot a timeline of each speaker's activity for audeo duration.
    """
    fig, ax = plt.subplots(figsize=(10, 4), dpi=400)
    speakers = sorted({speaker for _, _, speaker in diarization.itertracks(yield_label=True)})
    colors = {speaker: palette[i % len(palette)] for i, speaker in enumerate(speakers)}
    line_height = 6
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        ax.plot([turn.start, turn.end],
                [speakers.index(speaker)] * 2,
                color=colors[speaker],
                linewidth=line_height,
                alpha=0.9,
                solid_capstyle='round')
    ax.set_yticks(range(len(speakers)))
    ax.set_yticklabels([f"Speaker {s}" for s in speakers], fontsize=9)
    ax.set_xlim(0, total_duration)
    ax.set_xlabel("Time (seconds)", fontsize=10)
    ax.set_title("Speaker Activity Timeline", fontsize=11)
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color('#cccccc')
    ax.grid(True, axis='x', linestyle='-', linewidth=0.5, color='#eeeeee')
    plt.tight_layout()
    return fig

def generate_histogram(csv_file):
    """
    Create a histogram of word counts per 5-second bucket with color-coded sentiment.
    """
    df = pd.read_csv(csv_file)
    df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
    def make_label(start):
        minutes = int(start) // 60
        seconds = int(start) % 60
        end_sec = int(start) + 5
        end_minutes = end_sec // 60
        end_seconds = end_sec % 60
        return f"{minutes:02d}:{seconds:02d}-{end_minutes:02d}:{end_seconds:02d}"
    df['bucket_label'] = df['start'].apply(make_label)
    bar_colors = df['sentiment'].map(sentiment_colors)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df['bucket_label'], df['word_count'], color=bar_colors)
    ax.set_xlabel("Time Bucket (mm:ss-mm:ss)", fontsize=10)
    ax.set_ylabel("Word Count", fontsize=10)
    ax.set_title("Word Count per 5-Second Segment", fontsize=11)
    plt.xticks(rotation=90, fontsize=8)
    patches = [mpatches.Patch(color=sentiment_colors[s], label=s) for s in ["Positive", "Negative", "Neutral"]]
    ax.legend(handles=patches, fontsize=8)
    plt.tight_layout()
    return fig

def generate_sentiment_plot(csv_file):
    """
    Bar chart of overall sentiment counts.
    """
    df = pd.read_csv(csv_file)
    counts = df['sentiment'].value_counts().reindex(["Positive", "Negative", "Neutral"], fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind='bar', color=[sentiment_colors[s] for s in counts.index], ax=ax)
    ax.set_title("Sentiment Distribution", fontsize=11)
    plt.xticks(rotation=0, fontsize=9)
    plt.tight_layout()
    return fig

def plot_engagement_metrics(diarization, total_duration):
    """
    Plot talk time and turn counts per speaker.
    """
    speaker_stats = defaultdict(lambda: {'talk_time': 0, 'turns': 0})
    for track, _, speaker in diarization.itertracks(yield_label=True):
        speaker_stats[speaker]['talk_time'] += track.end - track.start
        speaker_stats[speaker]['turns'] += 1
    speakers = [f"Speaker {s}" for s in speaker_stats.keys()]
    talk_times = [v['talk_time'] for v in speaker_stats.values()]
    turn_counts = [v['turns'] for v in speaker_stats.values()]
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.barplot(y=speakers, x=talk_times, ax=ax[0], palette="Blues_d")
    sns.barplot(y=speakers, x=turn_counts, ax=ax[1], palette="Greens_d")
    ax[0].set_title('Talk Time (seconds)', fontsize=10)
    ax[1].set_title('Turn Count', fontsize=10)
    plt.tight_layout()
    return fig

def plot_speaker_heatmap(diarization, total_duration, time_bin_size=5):
    """
    Heatmap of speaker activity in fixed time bins.
    Activity is described as total number of words uttered in that duration.
    """
    bins = np.arange(0, int(total_duration) + time_bin_size, time_bin_size)
    speakers = sorted({speaker for _, _, speaker in diarization.itertracks(yield_label=True)})
    activity_matrix = np.zeros((len(speakers), len(bins) - 1))

    for track, _, speaker in diarization.itertracks(yield_label=True):
        i = speakers.index(speaker)
        bin_idx_start = np.digitize(track.start, bins) - 1
        bin_idx_end = np.digitize(track.end, bins) - 1
        activity_matrix[i, bin_idx_start:bin_idx_end + 1] += 1

    x_labels = [f"{int(b // 60):02d}:{int(b % 60):02d}" for b in bins[:-1]]
    
    fig, ax = plt.subplots(figsize=(18, 4)) 
    sns.heatmap(activity_matrix, cmap="YlGnBu", xticklabels=x_labels, 
                yticklabels=[f"Speaker {s}" for s in speakers], ax=ax)

    ax.set_xlabel('Time (mm:ss)')
    ax.set_ylabel('Speakers')
    ax.set_title('Speaker Activity Heatmap (5-sec Bins)')

    ax.set_xticks(np.arange(0, len(x_labels), max(1, len(x_labels) // 20)))  # Show every nth label
    ax.set_xticklabels(x_labels[::max(1, len(x_labels) // 20)], rotation=45, ha="right")

    plt.tight_layout()
    return fig

def plot_interaction_network(diarization):
    """
    Network graph showing how often speakers follow each other.
    """
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
    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, node_size=1200, node_color='#ffc107', ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9, ax=ax)
    edges = G.edges(data=True)
    widths = [0.5 + data['weight'] for _, _, data in edges]
    nx.draw_networkx_edges(G, pos, width=widths, edge_color='#666666', arrowsize=20, ax=ax)
    edge_labels = {(u, v): d['weight'] for u, v, d in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black', font_size=8, ax=ax)
    ax.set_title('Speaker Interaction Network', fontsize=10)
    ax.axis('off')
    plt.tight_layout()
    return fig

def compute_speaker_segments(diarization, trans_model, audio_path):
    """
    Extract text for each speaker segment.
    """
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

def plot_total_word_counts(speaker_segments):
    """
    Bar plot of total word counts per speaker.
    """
    speakers = []
    total_words = []
    for speaker, segments in speaker_segments.items():
        count = sum(len(text.split()) for _, _, text, _ in segments)
        speakers.append(f"Speaker {speaker}")
        total_words.append(count)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=speakers, y=total_words, palette="rocket", ax=ax)
    ax.set_xlabel("Speaker", fontsize=9)
    ax.set_ylabel("Total Word Count", fontsize=9)
    ax.set_title("Total Word Count per Speaker", fontsize=10)
    plt.xticks(rotation=45, fontsize=8)
    plt.tight_layout()
    return fig

def plot_average_turn_length(speaker_segments):
    """
    Bar plot of average turn length (in seconds) per speaker.
    """
    speakers = []
    avg_turn = []
    for speaker, segments in speaker_segments.items():
        lengths = [end - start for start, end, text, color in segments]
        avg = sum(lengths)/len(lengths) if lengths else 0
        speakers.append(f"Speaker {speaker}")
        avg_turn.append(avg)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=speakers, y=avg_turn, palette="magma", ax=ax)
    ax.set_xlabel("Speaker", fontsize=9)
    ax.set_ylabel("Avg Turn Length (s)", fontsize=9)
    ax.set_title("Average Turn Length per Speaker", fontsize=10)
    plt.xticks(rotation=45, fontsize=8)
    plt.tight_layout()
    return fig

def plot_cumulative_talk_time(speaker_segments, total_duration, bin_size=10):
    """
    Line plot of how each speaker's talk time accumulates over the audio duration.
    """
    speakers = sorted(speaker_segments.keys())
    bins = np.arange(0, total_duration+bin_size, bin_size)
    cumulative_data = {speaker: np.zeros(len(bins)-1) for speaker in speakers}
    for speaker, segments in speaker_segments.items():
        for start, end, text, color in segments:
            bin_idx_start = np.digitize(start, bins) - 1
            bin_idx_end = np.digitize(end, bins) - 1
            for i in range(bin_idx_start, bin_idx_end+1):
                cumulative_data[speaker][i] += min(end, bins[i+1]) - max(start, bins[i])
    fig, ax = plt.subplots(figsize=(10, 5))
    for speaker in speakers:
        ax.plot(bins[:-1], np.cumsum(cumulative_data[speaker]), label=f"Speaker {speaker}")
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Cumulative Talk Time (s)", fontsize=9)
    ax.set_title("Cumulative Talk Time per Speaker", fontsize=10)
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig

def FUNCTION1_results(trans_model, audio_path, base_name):
    rows = []
    current_window_start = 0
    current_window_end = 5
    current_text = ""
    segments, _ = trans_model.transcribe(audio_path, beam_size=2, word_timestamps=True)

    sentiment_idx = 0
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
                        "sentiment": analyze_sentiment(sent_model, current_text.strip())
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
            "sentiment": analyze_sentiment(sent_model, current_text.strip())
        })

    df = pd.DataFrame(rows)
    csv_file_path = os.path.join(csv_files, base_name + ".csv")
    df.to_csv(csv_file_path, index=False)
    return csv_file_path

def FUNCTION2_plot_figures(diarization, csv_file_path, audio_duration, trans_model, audio_path, base_name):
    """
    Generate and display various communication metric plots.
    """
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

    timeline_fig.show()
    histogram_fig.show()
    sentiment_fig.show()
    engagement_fig.show()
    heatmap_fig.show()
    network_fig.show()
    total_words_fig.show()
    avg_turn_fig.show()
    cumulative_fig.show()

    def save_figure(fig, figs_dir, base_name, fig_name):
        os.makedirs(figs_dir, exist_ok=True)
        save_path = os.path.join(figs_dir, f"{base_name}_{fig_name}.png")
        fig.savefig(save_path)

    save_figure(timeline_fig, figs_dir, base_name, "timeline")
    save_figure(histogram_fig, figs_dir, base_name, "histogram")
    save_figure(sentiment_fig, figs_dir, base_name, "sentiment")
    save_figure(engagement_fig, figs_dir, base_name, "engagement")
    save_figure(heatmap_fig, figs_dir, base_name, "heatmap")
    save_figure(network_fig, figs_dir, base_name, "network")
    save_figure(total_words_fig, figs_dir, base_name, "total_words")
    save_figure(avg_turn_fig, figs_dir, base_name, "avg_turn")
    save_figure(cumulative_fig, figs_dir, base_name, "cumulative")
    print(f"Saved all figures to {figs_dir} directory.")

trans_model = load_trans_model()
diar_model = load_diar_model()
sent_model = load_sent_model()

def main(videos_path):
    for vid in os.listdir(videos_path):
        video_path = os.path.join(videos_path, vid)
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        audio_path = os.path.join(exported_audio_dir, f"{base_name}.wav")
        audio = AudioSegment.from_file(video_path)
        audio.export(audio_path, format="wav")
        diarization = get_diarization(audio_path)
        csv_file_path = FUNCTION1_results(trans_model, audio_path, base_name)
        FUNCTION2_plot_figures(diarization, csv_file_path, len(audio)/1000, trans_model, audio_path, base_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process all video files in a folder.")
    parser.add_argument('video_folder', nargs='?', default='vids', help="Path to the folder containing video files (default: vids)")
    args = parser.parse_args()
    main(args.video_folder)