from dataclasses import dataclass, field
from typing import List


@dataclass
class ProjectImage:
    image_path: str = None
    width: int = None
    height: int = None
    zoom: float = None
    reverse_ratio: float = None
    time: float = None
    start: float = None
    stop: float = None
    frames: List["ProjectFrame"] = field(default_factory=list)


@dataclass
class ProjectFrame:
    idx: int = None
    image: ProjectImage = None
    time: float = None
    start: float = None
    end: float = None
