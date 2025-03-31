import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import seaborn as sns
import networkx as nx
from collections import defaultdict
import pandas as pd
import os
import numpy as np
import warnings
import logging
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

logging.getLogger("matplotlib.category").setLevel(logging.ERROR)
logging.getLogger("seaborn").setLevel(logging.ERROR)

figs_dir = "outputs/figs/"

sns.set_theme(style="whitegrid")
sentiment_colors = {"Positive": "#4CAF50", "Negative": "#F44336", "Neutral": "#9E9E9E"}
cmap = LinearSegmentedColormap.from_list("custom_heatmap", ["#f7fbff", "#08306b"])
palette = sns.color_palette("husl", 8)

def plot_speaker_timeline(diarization, total_duration):
    fig, ax = plt.subplots(figsize=(10, 4), dpi=400)
    speakers = sorted({speaker for _, _, speaker in diarization.itertracks(yield_label=True)})
    colors = {speaker: palette[i % len(palette)] for i, speaker in enumerate(speakers)}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        ax.plot([turn.start, turn.end], [speakers.index(speaker)] * 2, 
                color=colors[speaker], linewidth=6, alpha=0.9, solid_capstyle='round')
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
    df = pd.read_csv(csv_file)
    df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
    df['bucket_label'] = df['start'].apply(lambda s: f"{int(s)//60:02d}:{int(s)%60:02d}-{(int(s)+5)//60:02d}:{(int(s)+5)%60:02d}")
    bar_colors = df['sentiment'].map(sentiment_colors)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(df)), df['word_count'], color=bar_colors)
    ax.set_xlabel("Time Bucket (mm:ss-mm:ss)", fontsize=10)
    ax.set_ylabel("Word Count", fontsize=10)
    ax.set_title("Word Count", fontsize=11)
    tick_positions = list(range(0, len(df), 3))
    tick_labels = [df['bucket_label'].iloc[i] for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=8)
    
    patches = [mpatches.Patch(color=sentiment_colors[s], label=s) for s in ["Positive", "Negative", "Neutral"]]
    ax.legend(handles=patches, fontsize=8)
    plt.tight_layout()
    return fig


def generate_sentiment_plot(csv_file):
    df = pd.read_csv(csv_file)
    counts = df['sentiment'].value_counts().reindex(["Positive", "Negative", "Neutral"], fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind='bar', color=[sentiment_colors[s] for s in counts.index], ax=ax)
    ax.set_title("Sentiment Distribution", fontsize=11)
    plt.xticks(rotation=0, fontsize=9)
    plt.tight_layout()
    return fig

def plot_engagement_metrics(diarization, total_duration):
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
    ax.set_xlabel('Time (mm:ss)', fontsize=9)
    ax.set_ylabel('Speakers', fontsize=9)
    ax.set_title('Speaker Activity Heatmap', fontsize=10)
    ax.set_xticks(np.arange(0, len(x_labels), max(1, len(x_labels) // 20)))
    ax.set_xticklabels(x_labels[::max(1, len(x_labels) // 20)], rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
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

def plot_total_word_counts(speaker_segments):
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

def FUNCTION2_plot_figures(diarization, csv_file_path, audio_duration, trans_model, audio_path, base_name, speaker_segments):
    timeline_fig = plot_speaker_timeline(diarization, audio_duration)
    histogram_fig = generate_histogram(csv_file_path)
    sentiment_fig = generate_sentiment_plot(csv_file_path)
    engagement_fig = plot_engagement_metrics(diarization, audio_duration)
    heatmap_fig = plot_speaker_heatmap(diarization, audio_duration, time_bin_size=5)
    network_fig = plot_interaction_network(diarization)
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

    def save_figure(fig, fig_name):
        save_path = os.path.join(figs_dir, f"{base_name}_{fig_name}.png")
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        # print_progress(f"Saved {fig_name} figure to {save_path}", color=BLUE)

    save_figure(timeline_fig, "timeline")
    save_figure(histogram_fig, "histogram")
    save_figure(sentiment_fig, "sentiment")
    save_figure(engagement_fig, "engagement")
    save_figure(heatmap_fig, "heatmap")
    save_figure(network_fig, "network")
    save_figure(total_words_fig, "total_words")
    save_figure(avg_turn_fig, "avg_turn")
    save_figure(cumulative_fig, "cumulative")
