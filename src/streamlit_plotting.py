import os
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
import networkx as nx
from collections import defaultdict
from pydub import AudioSegment

sentiment_colors = {"Positive": "#4CAF50", "Negative": "#F44336", "Neutral": "#9E9E9E"}

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

