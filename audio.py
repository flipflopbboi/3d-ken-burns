from dataclasses import dataclass
from datetime import timedelta
from typing import List

import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from librosa import get_duration
from matplotlib import patches, cm, colors
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from sklearn.cluster import AgglomerativeClustering

from helpers.dict import list_to_dict_list


@dataclass
class ProjectAudio:
    file: str
    signal: np.ndarray
    sampling_rate: float
    duration: timedelta
    time_cluster: List["AudioTimeCluster"]


@dataclass
class AudioCluster:
    idx: int
    id: str = None
    intensity: float = None
    cover: float = None


@dataclass
class AudioTimeCluster:
    start: timedelta
    end: timedelta
    start_point: int
    end_point: int
    cluster: AudioCluster

    @property
    def duration(self) -> timedelta:
        return self.end - self.start


def cluster_audio(
    audio_file: str, n_clusters: int = 6, plot: bool = False
) -> List[AudioTimeCluster]:
    y, sr = librosa.load(audio_file)
    tempogram: np.ndarray = librosa.feature.tempogram(y=y, sr=sr)

    n_points = tempogram.shape[1]
    secs_per_point = timedelta(seconds=get_duration(y, sr)) / n_points

    clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(
        tempogram.transpose()
    )
    labels: np.ndarray = clustering.labels_

    clusters = [AudioCluster(idx=idx) for idx in set(labels)]

    points_of_change: np.ndarray = np.nonzero(np.diff(labels))[0].tolist()
    # Add beginning and end
    points_of_change: np.ndarray = [0] + points_of_change + [n_points - 1]

    time_clusters: List[AudioTimeCluster] = []
    for start_point, end_point in zip(points_of_change, points_of_change[1:]):
        start_point = start_point + 1
        time_clusters.append(
            AudioTimeCluster(
                start=start_point * secs_per_point,
                end=end_point * secs_per_point,
                start_point=start_point,
                end_point=end_point,
                cluster=clusters[labels[end_point]],
            )
        )

    for cluster, times in list_to_dict_list(time_clusters, key_func=lambda t: t.cluster):
        duration: timedelta = sum(t.end - t.start for t in times)
        cluster.cover = duration/timedelta(seconds=get_duration(y, sr))
        cluster.intensity = (tempogram[:, start_point:end_point].sum() / tempogram.sum(),)

    if plot:
        min_intensity: float = min(t.intensity for t in time_clusters)
        max_intensity: float = max(t.intensity for t in time_clusters)
        norm = colors.Normalize(vmin=min_intensity, vmax=max_intensity, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap="hot")

        fig, ax = plt.subplots(1)
        for time_cluster in time_clusters:

            rects = [
                Rectangle(
                    xy=(time_cluster.start_point, time_cluster.cluster_idx),
                    width=time_cluster.end_point - time_cluster.start_point,
                    height=1,
                    color="y" if time_cluster.intensity < 0.02 else "r",
                    # facecolor=mapper.to_rgba(time_cluster.intensity),
                )
            ]
            # rects.append(rect)
            pc = PatchCollection(
                rects,
                alpha=0.5,
                edgecolor="black",
                color=mapper.to_rgba(time_cluster.intensity),
            )
            ax.add_collection(pc)
        ax.set_xlim((0, n_points))
        ax.set_ylim((0, n_clusters))

        # Add x-axis ticks
        plt.xticks(np.arange(0, n_points, timedelta(seconds=20) / secs_per_point))
        n_xtick_labels = len(ax.get_xticklabels())
        timedeltas: List[timedelta] = [
            timedelta(seconds=20 * tick_idx) for tick_idx in range(n_xtick_labels)
        ]
        str_labels: List[str] = [":".join(str(td).split(":")[1:]) for td in timedeltas]
        ax.set_xticklabels(str_labels)

        # Add y-axis ticks
        plt.yticks(np.arange(0, n_clusters, 1))
        str_labels: List[str] = [
            f"Cluster #{time_cluster.cluster_idx} [{time_cluster.intensity:3.2f}%]"
            for time_cluster in time_clusters
        ]
        ax.set_yticklabels(str_labels)

        plt.grid()
        plt.show()

    return time_clusters


def plot_all(audio_file: str) -> None:
    y, sr = librosa.load(audio_file)
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

if __name__ == "__main__":
    AUDIO_FILE = "./audio/peaches.mp3"
    cluster_audio(audio_file=AUDIO_FILE, n_clusters=6, plot=True)
