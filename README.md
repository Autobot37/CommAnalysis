# Video Communication Analysis Pipeline

## Overview

This Python script provides a comprehensive pipeline for analyzing communication patterns within video files. It extracts audio, identifies speakers (diarization), transcribes speech, performs sentiment analysis and intent classification on the transcriptions, and generates various analytical visualizations and reports. The analysis focuses specifically on intents relevant to a driving context.

## Core Functionalities

1.  **Data Preparation:**
    * Automatically checks for input video data in the `data/gsocvideos/` directory.
    * If the directory is empty, it downloads sample video data from a predefined Google Drive folder.
    * Extracts audio from input video files and saves it as WAV format in `outputs/exported_audio/`.
    * Creates necessary directories for caching, temporary files, and outputs.

2.  **Speaker Diarization:**
    * Utilizes the `pyannote/speaker-diarization-3.1` model via the `pyannote.audio` library to determine *who* spoke *when* in the audio.
    * Results are cached (`cache/diarization_cache/`) to speed up subsequent runs.

3.  **Speech Transcription:**
    * Uses the `faster-whisper` library (with a specified model size, e.g., `small.en`) to transcribe the audio content.
    * Supports word-level timestamps for detailed analysis.
    * Transcription results for full audio and individual speaker segments are cached (`cache/transcription_cache/`).

4.  **Sentiment Analysis:**
    * Employs a weighted combination of two sentiment analysis models:
        * **NLTK VADER:** Lexicon-based approach.
        * **CardiffNLP RoBERTa:** Transformer-based model fine-tuned for sentiment (`cardiffnlp/twitter-roberta-base-sentiment-latest`). Handles long text segments by chunking.
    * Classifies text segments as "Positive", "Negative", or "Neutral".

5.  **Intent Classification:**
    * Uses a `CrossEncoder` model (`cross-encoder/ms-marco-MiniLM-L6-v2`) to classify text segments according to predefined driving-related intents (e.g., `change_speed`, `navigation`, `weather_update`, `accident_warning`).
    * Predicts the most likely intent for a given text segment.

6.  **Temporal Analysis:**
    * Analyzes the conversation over time using one of two strategies (configurable via command-line argument `--st`):
        * **`chunk`:** Divides the transcription into fixed-time chunks (e.g., 5 seconds).
        * **`sentence`:** Divides the transcription based on sentence boundaries.
    * Applies a sliding window (`--wsz` argument) to incorporate context from previous chunks/sentences for more robust sentiment and intent analysis.
    * Generates a primary CSV file (`outputs/csv_files/{video_base_name}.csv`) containing timestamps, text segments, sentiment, and intent for each analyzed window.

7.  **Speaker-Specific Analysis:**
    * Segments the audio based on diarization results.
    * Transcribes and analyzes the text for each speaker individually.
    * Applies context windowing within each speaker's turns.
    * Generates separate CSV files for each speaker (`outputs/csv_files/{video_base_name}_SPEAKER_{id}.csv`).

8.  **Visualization & Reporting:**
    * Generates a suite of plots using Matplotlib and Seaborn, saved to `outputs/figs/`:
        * `timeline`: Speaker activity over time.
        * `histogram`: Word count per time bucket, colored by sentiment.
        * `sentiment`: Overall distribution of sentiments.
        * `engagement`: Bar charts of total talk time and turn count per speaker.
        * `heatmap`: Speaker activity intensity across time bins.
        * `network`: Speaker interaction turn-taking graph.
        * `total_words`: Total word count per speaker.
        * `avg_turn`: Average turn length per speaker.
        * `cumulative`: Cumulative talk time per speaker over the conversation duration.

## Requirements

* Python 3.x
* PyTorch (with CUDA support recommended for GPU acceleration)
* Key Python Libraries: `pyannote.audio`, `faster-whisper`, `transformers`, `sentence-transformers`, `nltk`, `pydub`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `networkx`, `huggingface_hub`, `gdown`, `tqdm`.
    * *(Note: You should generate a `requirements.txt` file from the imports)*
* Hugging Face Hub Token (for accessing `pyannote` model)
* NLTK VADER Lexicon (downloaded automatically by the script)

## Setup

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Hugging Face Login:**
    * Obtain a Hugging Face Hub token with access permissions.
    * Replace the placeholder `"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"` in the script with your actual token, or set it up via the `huggingface-cli login` command.
3.  **Place Videos:** Put your input video files into the `data/gsocvideos/` directory (or leave it empty to download sample data).
4.  **Directory Structure:** The script will automatically create the necessary `cache`, `outputs`, and `tempdata` directories if they don't exist.

## Usage

Run the script from the command line:

```bash
python your_script_name.py [folder] [--st strategy] [--wsz window_size]