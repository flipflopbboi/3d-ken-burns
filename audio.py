from dataclasses import dataclass
from datetime import timedelta
from typing import List, Tuple, Dict

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from librosa import get_duration
from matplotlib import cm, colors
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from sklearn.cluster import AgglomerativeClustering

from config import N_AUDIO_CLUSTERS
from helpers.cache import property_cached
from helpers.datetime import sum_timedeltas
from helpers.dict import list_to_dict_list


@dataclass
class AudioCluster:
    name: str
    idx: int
    intensity: float = None
    cover: float = None
    duration: timedelta = None
    count: int = None

    def __hash__(self):
        return hash(self.name)


@dataclass
class ProjectAudio:
    file: str

    def __post_init__(self):
        r: Tuple[np.ndarray, float] = librosa.load(self.file)
        self.signal = r[0]
        self.sampling_rate = r[1]
        self.n_samples: int = len(self.signal)
        self.duration: timedelta = timedelta(
            seconds=get_duration(self.signal, self.sampling_rate)
        )
        self.n_points: int = self.tempogram.shape[1]  # refers to the tempogram samples
        self.secs_per_point: timedelta = self.duration / self.n_points
        self.time_clusters = self.get_time_clusters(n_clusters=N_AUDIO_CLUSTERS)
        self.clusters = sorted(
            list(set(t.cluster for t in self.time_clusters)), key=lambda c: c.name
        )
        self.n_clusters = len(self.clusters)

    @property_cached
    def tempogram(self) -> np.ndarray:
        return librosa.feature.tempogram(y=self.signal, sr=self.sampling_rate)

    def get_time_clusters(self, n_clusters: int = 6) -> List["AudioTimeCluster"]:

        clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(
            self.tempogram.transpose()
        )
        labels: np.ndarray = clustering.labels_

        clusters = [AudioCluster(name=idx, idx=idx) for idx in set(labels)]

        points_of_change: List[int] = np.nonzero(np.diff(labels))[0].tolist()
        points_of_change: List[int] = [0] + points_of_change + [
            self.tempogram.shape[1] - 1
        ]

        time_clusters: List[AudioTimeCluster] = []
        for start_point, end_point in zip(points_of_change, points_of_change[1:]):
            start_point = start_point + 1
            time_clusters.append(
                AudioTimeCluster(
                    start=start_point * self.secs_per_point,
                    end=end_point * self.secs_per_point,
                    start_point=start_point,
                    end_point=end_point,
                    cluster=clusters[labels[end_point]],
                )
            )

        times_by_cluster: Dict[
            AudioCluster, List[AudioTimeCluster]
        ] = list_to_dict_list(time_clusters, key_func=lambda t: t.cluster)

        for cluster, times in times_by_cluster.items():
            cluster.duration: timedelta = sum_timedeltas(
                [(t.end - t.start) for t in times]
            )
            cluster.count: int = len(time_clusters)
            cluster.cover = cluster.duration / self.duration
            cluster.intensity = (
                sum(
                    self.tempogram[:, t.start_point : t.end_point].sum()
                    for t in times
                )
                / self.tempogram.sum()
            )
        return time_clusters

    def plot_clusters(self) -> None:
        """
        Plot all time clusters.
        """
        min_intensity: float = min(t.cluster.intensity for t in self.time_clusters)
        max_intensity: float = max(t.cluster.intensity for t in self.time_clusters)
        norm = colors.Normalize(vmin=min_intensity, vmax=max_intensity, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap="hot")

        fig, ax = plt.subplots(1, figsize=(12, 6), dpi=300)
        for time_cluster in self.time_clusters:
            rects = [
                Rectangle(
                    xy=(time_cluster.start_point, time_cluster.cluster.idx),
                    width=time_cluster.end_point - time_cluster.start_point,
                    height=1,
                    edgecolor="b",
                )
            ]
            pc = PatchCollection(
                rects,
                alpha=0.5,
                edgecolor="b",
                color=mapper.to_rgba(time_cluster.cluster.intensity),
            )
            ax.add_collection(pc)
        ax.set_xlim((0, self.n_points))
        ax.set_ylim((0, self.n_clusters))

        # Add x-axis ticks
        plt.xticks(
            np.arange(0, self.n_points, timedelta(seconds=20) / self.secs_per_point)
        )
        n_xtick_labels = len(ax.get_xticklabels())
        timedeltas: List[timedelta] = [
            timedelta(seconds=20 * tick_idx) for tick_idx in range(n_xtick_labels)
        ]
        str_labels: List[str] = [":".join(str(td).split(":")[1:]) for td in timedeltas]
        ax.set_xticklabels(str_labels)

        # Add y-axis ticks
        plt.yticks(np.arange(0, self.n_clusters, 1))
        str_labels: List[str] = [
            f"Cluster #{cluster.idx} [{cluster.intensity*100:5.1f}%]"
            for cluster in self.clusters
        ]
        ax.set_yticklabels(str_labels)
        # plt.grid()
        # plt.figure(figsize=(12, 6), dpi=300)
        plt.show()

    def get_time_list_from_audio_beats(self, audio_file: str) -> List[float]:
        if self.file is None:
            return []
        y, sr = librosa.load(audio_file)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        beat_times: np.ndarray = librosa.frames_to_time(beats, sr=sr)
        time_list = np.diff(beat_times)
        np.append(time_list, [1])
        return time_list.tolist()


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

    audio = ProjectAudio(file=AUDIO_FILE)
    audio.get_time_clusters(n_clusters=6)
    audio.plot_clusters()
