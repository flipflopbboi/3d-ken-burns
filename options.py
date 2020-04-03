from dataclasses import dataclass


@dataclass
class ProjectOptions:
    input: str
    output: str
    folder: str
    video: str
    zoom: float
    shift: float
    start: float
    stop: float
    time: float
    width: float
    height: float
    audio: str
    face_zoom: bool = False
    reverse: bool = False
    step_factor: bool = False
    random_order: bool = False
    random_zoom: bool = False
    jitter: bool = False



    # def print_summary(self):
