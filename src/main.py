from pyannote.audio import Pipeline
import torch
from faster_whisper import WhisperModel
import pickle
from transformers import pipeline
from pydub import AudioSegment
import warnings
import logging
import argparse

from src.plotting import * 

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
SAVE_CACHE = False

videos_path = "data/gsocvideos/"
diarization_dir = "cache/diarization_cache/"
cache_dir = "cache/transcription_cache/"
exported_audio_dir = "outputs/exported_audio/"
csv_files = "outputs/csv_files/"
temp_audio = "tempdata/temp_audio/"

for d in [videos_path, exported_audio_dir, diarization_dir, csv_files, cache_dir, temp_audio, figs_dir]:
    os.makedirs(d, exist_ok=True)

if(len(os.listdir(videos_path)) == 0):
    import gdown
    print("input data is empty downloading video from drive.")
    folder_url = "https://drive.google.com/drive/folders/1clnoqARUaLfR-fC42w8WkwYmCqfZtoN5"
    gdown.download_folder(folder_url, quiet=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

def print_progress(message, color=GREEN):
    print(f"{color}{message}{RESET}")

def load_sent_model():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", device=0 if device=="cuda" else -1)

def load_trans_model():
    model_size = "small.en"
    return WhisperModel(model_size, device=device, compute_type="float16" if device=="cuda" else "int8")

def load_diar_model():
    pipeline_model = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token="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")
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

def FUNCTION1_results(trans_model, audio_path, base_name):
    rows = []
    current_window_start = 0
    current_window_end = 5
    current_text = ""
    print_progress("[+] transcribing text", BLUE)
    segments, _ = trans_model.transcribe(audio_path, beam_size=2, word_timestamps=True)
    print_progress("[+] transcribing completed", GREEN)
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

def main(videos_folder):
    for vid in os.listdir(videos_folder):
        video_path = os.path.join(videos_folder, vid)
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        audio_path = os.path.join(exported_audio_dir, f"{base_name}.wav")
        audio = AudioSegment.from_file(video_path)
        audio.export(audio_path, format="wav")
        print_progress(f"[+] processing video: {vid}", color=YELLOW)
        print_progress("[+] processing diarization", BLUE)
        diarization = get_diarization(audio_path)
        print_progress("[+] diarization completed", GREEN)
        print_progress("[+] generating csv file with sentiment analysis.", BLUE)
        csv_file_path = FUNCTION1_results(trans_model, audio_path, base_name)
        print_progress("[+] csv file generated", GREEN)
        print_progress("[+] computing speaker segments.", BLUE)
        speaker_segments = compute_speaker_segments(diarization, trans_model, audio_path)
        print_progress("[+] speaker segmented computed..", GREEN)
        print_progress("[+] plotting figures.", BLUE)
        FUNCTION2_plot_figures(diarization, csv_file_path, len(audio)/1000, trans_model, audio_path, base_name, speaker_segments)
        print_progress("[+] all figures plotted.", GREEN)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process video files for communication analysis.")
    parser.add_argument("video_folder", nargs='?', default=videos_path, help="Folder containing video files")
    args = parser.parse_args()
    trans_model = load_trans_model()
    print_progress("[+] transcription model loaded.", GREEN)
    diar_model = load_diar_model()
    print_progress("[+] diarization model loaded.", GREEN)
    sent_model = load_sent_model()
    print_progress("[+] sentiment model loaded.", GREEN)
    main(args.video_folder)
