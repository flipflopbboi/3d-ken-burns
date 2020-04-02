from dataclasses import dataclass


@dataclass
class ProjectImage:
    image_path: str
    zoom: float = None
    reverse_ratio: float = None
    time: float = None
    start: float = None
    stop: float = None
