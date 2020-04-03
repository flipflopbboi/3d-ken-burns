from dataclasses import dataclass
from datetime import timedelta
from typing import List

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from librosa import get_duration
from sklearn.cluster import AgglomerativeClustering


@dataclass
class AudioTimeCluster:
    start: timedelta
    end: timedelta
    cluster_id: str

    @classmethod
    def from_sample_times(
        cls, start_point: int, end_point: int, cluster_id: str, points_per_sec: float
    ) -> "AudioTimeCluster":
        return AudioTimeCluster(
            start=timedelta(seconds=start_point * points_per_sec),
            end=timedelta(seconds=end_point * points_per_sec),
            cluster_id=cluster_id,
        )


def cluster_audio(audio_file: str, plot: bool = False) -> List[AudioTimeCluster]:
    y, sr = librosa.load(audio_file)
    tempogram: np.ndarray = librosa.feature.tempogram(y=y, sr=sr)
    N_CLUSTERS = 8

    n_points = tempogram.shape[1]
    points_per_sec = timedelta(get_duration(y, sr)) / n_points

    clustering = AgglomerativeClustering(n_clusters=N_CLUSTERS).fit(
        tempogram.transpose()
    )
    labels: np.ndarray = clustering.labels_
    points_of_change: np.ndarray = np.nonzero(np.diff(labels))[0].tolist()
    # Add beginning and end
    points_of_change: np.ndarray = [0] + points_of_change + [n_points - 1]

    clusters: List[AudioTimeCluster] = []
    for start, end in zip(points_of_change, points_of_change[1:]):
        start = start + 1
        clusters.append(
            AudioTimeCluster.from_sample_times(
                start_point=start,
                end_point=end,
                cluster_id=labels[end],
                points_per_sec=points_per_sec,
            )
        )

    if plot:
        plt.scatter(x=range(n_points), y=labels)
        plt.show()

        fig = plt.figure(figsize=(20, 6), facecolor="w")
        ax = plt.imshow(tempogram, cmap="hot", interpolation="nearest", aspect="auto")
        plt.title("Tempogram")
        plt.xlabel("Time")
        plt.ylabel("BPM")

        for i in points_of_change:
            plt.axvline(x=i, color="g")
        plt.show()

    return clusters


def plot_all():
    plt.figure(figsize=(20, 12), facecolor="w")
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    plt.subplot(4, 2, 1)
    librosa.display.specshow(D, y_axis="linear")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Linear-frequency power spectrogram")

    plt.subplot(4, 2, 2)
    librosa.display.specshow(D, y_axis="log")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Log-frequency power spectrogram")

    CQT = librosa.amplitude_to_db(np.abs(librosa.cqt(y, sr=sr)), ref=np.max)
    plt.subplot(4, 2, 3)
    librosa.display.specshow(CQT, y_axis="cqt_note")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Constant-Q power spectrogram (note)")

    plt.subplot(4, 2, 4)
    librosa.display.specshow(CQT, y_axis="cqt_hz")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Constant-Q power spectrogram (Hz)")

    C = librosa.feature.chroma_cqt(y=y, sr=sr)
    plt.subplot(4, 2, 5)
    librosa.display.specshow(C, y_axis="chroma")
    plt.colorbar()
    plt.title("Chromagram")

    plt.subplot(4, 2, 6)
    librosa.display.specshow(D, cmap="gray_r", y_axis="linear")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Linear power spectrogram (grayscale)")

    plt.subplot(4, 2, 7)
    librosa.display.specshow(D, x_axis="time", y_axis="log")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Log power spectrogram")

    plt.subplot(4, 2, 8)
    Tgram = librosa.feature.tempogram(y=y, sr=sr)
    librosa.display.specshow(Tgram, x_axis="time", y_axis="tempo")
    plt.colorbar()
    plt.title("Tempogram")


##########################################################

# if __name__ == "__main__":
#     AUDIO_FILE = "./audio/peaches.mp3"
#     y, sr = librosa.load(AUDIO_FILE)
#     cluster_audio(y, sr)
