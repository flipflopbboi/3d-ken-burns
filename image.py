import argparse
from dataclasses import dataclass, field
from typing import List

import cv2
import numpy as np


@dataclass
class ProjectImage:
    image_path: str = None
    array: np.ndarray = None
    width: int = None
    height: int = None
    w_to_h_ratio: float = None
    zoom: float = None
    reverse_ratio: float = None
    time: float = None
    start: float = None
    stop: float = None
    shift: float = None
    frames: List["ProjectFrame"] = field(default_factory=list)

    def set_array(self) -> None:
        self.array = cv2.imread(filename=self.image_path, flags=cv2.IMREAD_COLOR)
        self.width = self.array.shape[0]
        self.height = self.array.shape[1]
        self.w_to_h_ratio = self.width / self.height

    def resize(self, pixels: int = 1024) -> None:
        """

        """
        self.width = min(int(pixels * self.w_to_h_ratio), pixels)
        self.height = min(int(pixels * self.w_to_h_ratio), pixels)
        self.w_to_h_ratio = self.width / self.height
        self.array = cv2.resize(
            src=self.array,
            dsize=(self.width, self.height),
            fx=0.0,
            fy=0.0,
            interpolation=cv2.INTER_AREA,
        )

    @classmethod
    def from_args(cls, image_path: str, args: argparse.Namespace) -> "ProjectImage":
        return ProjectImage(
            image_path=image_path,
            zoom=args.zoom,
            time=args.time,
            start=args.start,
            stop=args.stop,
            shift=args.shift,
        )

    def get_face_coords(self):
        pass


@dataclass
class ProjectFrame:
    idx: int = None
    image: "ProjectImage" = None
    time: float = None
    start: float = None
    end: float = None
