import random
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import librosa
# from rvc.lib.tools.split_audio import process_audio, merge_audio
from rvc.lib.utils import load_audio_infer, load_embedding
import soundfile as sf
import os, sys
import torch
from rvc.configs.config import Config
import faiss
from rvc.lib.tools.split_audio import process_audio, merge_audio
import collections

def calculate_features(y, sr):
    stft = np.abs(librosa.stft(y))
    duration = librosa.get_duration(y=y, sr=sr)
    cent = librosa.feature.spectral_centroid(S=stft, sr=sr)[0]
    bw = librosa.feature.spectral_bandwidth(S=stft, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(S=stft, sr=sr)[0]
    return stft, duration, cent, bw, rolloff


def plot_title(title):
    plt.suptitle(title, fontsize=16, fontweight="bold")


def plot_spectrogram(y, sr, stft, duration, cmap="inferno"):
    plt.subplot(3, 1, 1)
    plt.imshow(
        librosa.amplitude_to_db(stft, ref=np.max),
        origin="lower",
        extent=[0, duration, 0, sr / 1000],
        aspect="auto",
        cmap=cmap,  # Change the colormap here
    )
    plt.colorbar(format="%+2.0f dB")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (kHz)")
    plt.title("Spectrogram")


def plot_waveform(y, sr, duration):
    plt.subplot(3, 1, 2)
    librosa.display.waveshow(y, sr=sr)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Waveform")


def plot_features(times, cent, bw, rolloff, duration):
    plt.plot(times, cent, label="Spectral Centroid (kHz)", color="b")
    plt.plot(times, bw, label="Spectral Bandwidth (kHz)", color="g")
    plt.plot(times, rolloff, label="Spectral Rolloff (kHz)", color="r")

    plt.xlabel("Time (s)")
    plt.title("Spectral Features")
    plt.legend()

def plot_feature_distance_comparison(raw_audios: list, audio_file_names: list, distance_points_per_audio_file: list, ):
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
    
    for idx, raw_audio in enumerate(raw_audios):
        random_color = random.choice(colors)
        colors.remove(random_color)
        number_of_points_in_plot = np.linspace(0, round(len(raw_audio)/16000, 3), distance_points_per_audio_file[idx].size)
        
        distance_mean = round(np.mean(distance_points_per_audio_file[idx]), 1)
        distance_std = round(np.std(distance_points_per_audio_file[idx]), 1)
    
        plt.plot(number_of_points_in_plot, distance_points_per_audio_file[idx], label=audio_file_names[idx] + f" (mean: {distance_mean} | std: {distance_std})", color=random_color)
    
    plt.ylabel("Feature distance")
    plt.xlabel("Time (s)")
    plt.title(f"Feature distance comparison ")
    plt.legend()

def plot_feature_distance(y, distances_points, file_name):
    plt.subplot(3, 1, 3)

    distance_mean = round(np.mean(distances_points), 1)
    distance_std = round(np.std(distances_points), 1)

    # for distance in distances:
    number_of_points_in_plot_1 = np.linspace(0, round(len(y)/16000, 3), distances_points.size)
    plt.plot(number_of_points_in_plot_1, distances_points, label=file_name, color="b")
    
    plt.ylabel("Feature distance")
    plt.xlabel("Time (s)")
    plt.title(f"Feature distance (mean: {distance_mean} | std: {distance_std})")
    plt.legend()

def analyze_audio(audio_file_path: str, index_path: str, save_plot_path: str):
    y, sr = librosa.load(audio_file_path, 16000)

    stft, duration, cent, bw, rolloff = calculate_features(y, sr)
    
    distances_points = get_distance(audio_file_path, index_path)
    
    plt.figure(figsize=(12, 10))

    plot_title("Audio Analysis" + " - " + audio_file_path.split("/")[-1])
    plot_spectrogram(y, sr, stft, duration)
    plot_waveform(y, sr, duration)
    plot_feature_distance(y, distances_points, audio_file_path.split('/')[-1])

    plt.tight_layout()

    plt.savefig(save_plot_path, bbox_inches="tight", dpi=300)
    plt.close()

    audio_info = f"""Sample Rate: {sr}\nDuration: {(
            str(round(duration, 2)) + " seconds"
            if duration < 60
            else str(round(duration / 60, 2)) + " minutes"
    )}\nNumber of Samples: {len(y)}\nBits per Sample: {librosa.get_samplerate(audio_file_path)}\nChannels: {"Mono (1)" if y.ndim == 1 else "Stereo (2)"}"""

    return audio_info, save_plot_path

def generate_comparison_plot(audio_paths: list, index_path: str):
    save_plot_path="logs/audio_comparison.png"
    
    raw_audios = []
    for audio_file in audio_paths:
        y, sr = librosa.load(audio_file, 16000) # Load an audio file as a floating point time series
        raw_audios.append(y)

    # pad shorter audio (we don't need it probably)
    # if (first_y.size > second_y.size):
    #     second_y = np.concatenate([second_y, np.zeros(first_y.size - second_y.size)])
    # if (second_y.size > first_y.size):
    #     first_y = np.concatenate([first_y, np.zeros(second_y.size - first_y.size)])

    plt.figure(figsize=(12, 3))

    plot_title("Audio Analysis")
    audio_file_names = [audio_file.split('/')[-1] for audio_file in audio_paths]
    distance_points_per_audio_file = [get_distance(audio_file, index_path) for audio_file in audio_paths]

    plot_feature_distance_comparison(raw_audios, audio_file_names, distance_points_per_audio_file)

    plt.tight_layout()

    plt.savefig(save_plot_path, bbox_inches="tight", dpi=300)
    plt.close()

    return save_plot_path

def get_distance(audio_path: str, index_path: str):
    
    config = Config()
    model = load_hubert_model(config)

    audio = load_audio(audio_path)
    audio_chunks, intervals = process_audio(audio, 16000)

    audio_intervals = {}
    for audio_chunk, interval in zip(audio_chunks, intervals):
        audio_intervals[tuple(interval)] = audio_chunk
    
    silence_intervals = get_silence_intervals(intervals, audio.size)
    for interval in silence_intervals:
        audio_intervals[tuple(interval)] = None

    audio_intervals = collections.OrderedDict(sorted(audio_intervals.items()))
    audio_points_per_feature = 320 # 16000/50 | 16000 is a sample rate per 1 second | there are 50 of 20ms chunks in 1 seconds 

    score = np.empty(0)
    for interval, values in audio_intervals.items():
        if values is not None:
            chunk_feats = prepare_feats(values, config)
            # extract features
            model_output = model(chunk_feats)
            out_feats = model_output["last_hidden_state"]
            index = faiss.read_index(index_path)
            npy = out_feats[0].cpu().detach().numpy()
            npy = npy.astype("float32") if config.is_half else npy
            chunk_score, ix = index.search(npy, k=8)
            
            min_chunk_score = np.min(chunk_score, axis=1)
            
            score = np.concatenate([score, min_chunk_score])

            # # moving average
            # kernel_value = 100
            # kernel = np.repeat(1, kernel_value)
            # padded_score = np.concatenate([np.zeros(len(kernel)//2), min_chunk_score, np.zeros(len(kernel)//2)])
            # score_moving_average = np.convolve(padded_score, kernel, "valid")
            # score_moving_average = score_moving_average[:-1] # Remove last element because valid add +1
            # score_moving_average = score_moving_average / kernel_value # Convolution is summing values without dividing them so we must divide it by number of kernel points
            # score = np.concatenate([score, score_moving_average])

        else:
            number_of_features = int((interval[1] - interval[0]) / audio_points_per_feature)
            # print(f"{number_of_features} features for interval {interval}")
            zeros = np.zeros(number_of_features)
            score = np.concatenate([score, zeros])

    return score

def load_hubert_model(config):
    hubert_model = load_embedding("contentvec", None)
    hubert_model.to(config.device)

    hubert_model = (
        hubert_model.half()
        if config.is_half
        else hubert_model.float()
    )
    hubert_model.eval()
    return hubert_model

def prepare_feats(audio, config):
    # audio = np.array(audio, dtype=np.int32)
    feats = (
        torch.from_numpy(audio).half()
        if config.is_half
        else torch.from_numpy(audio).float()
    )
    feats = feats.mean(-1) if feats.dim() == 2 else feats
    assert feats.dim() == 1, feats.dim()
    feats = feats.view(1, -1).to(config.device)
    return feats

def load_audio(audio_path):
    audio_path = audio_path.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"File not found: {audio_path}")
    audio, sr = librosa.load(audio_path, 16000)
    
    return audio

def get_silence_intervals(intervals, audio_size):
    silence_intervals = []

    if intervals[0][0] != 0: 
        silence_intervals.append([0, intervals[0][0]]) # first interval
    
    for idx, interval in enumerate(intervals):
        if idx != 0 and idx < len(intervals):
            silence_intervals.append([intervals[idx - 1][1], interval[0]])
    
    if intervals[len(intervals) - 1][1] != audio_size:
        silence_intervals.append([intervals[len(intervals) - 1][1], audio_size]) # last interval

    return silence_intervals