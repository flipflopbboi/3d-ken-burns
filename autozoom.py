#!/usr/bin/env python
import itertools
from dataclasses import dataclass

import librosa
import argparse
import pathlib
from typing import List, Dict, Tuple, Iterable
import pprint
import numpy as np

import torch
import torchvision
from imageio import imread
import base64
import cupy
import cv2
import flask
import getopt
import gevent
import gevent.pywsgi
import glob
import h5py
import io
import math
import moviepy
import moviepy.editor
import numpy
import os
import random
import re
import scipy
import scipy.io
import shutil
import sys
import tempfile
import time
import urllib
import zipfile

##########################################################
from more_itertools import chunked
from moviepy.video.compositing.concatenate import concatenate_videoclips
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from tqdm import tqdm

from config import FPS, DEFAULT_BORDER, N_IMAGES_PER_CHUNK
from helpers.logging import print_line, formatted_print, Color, print_success
from helpers.numeric import split_int_in_two
from image import ProjectImage

assert (
    int(str("").join(torch.__version__.split(".")[0:2])) >= 12
)  # requires at least pytorch version 1.2.0

torch.set_grad_enabled(
    False
)  # make sure to not compute gradients for computational performance

torch.backends.cudnn.enabled = (
    True
)  # make sure to use cudnn for computational performance

##########################################################

objCommon = {}

exec(open("./common.py", "r").read())

exec(open("./models/disparity-estimation.py", "r").read())
exec(open("./models/disparity-adjustment.py", "r").read())
exec(open("./models/disparity-refinement.py", "r").read())
exec(open("./models/pointcloud-inpainting.py", "r").read())

##########################################################


def parse_args(verbose: bool = True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", type=str, required=False, help="Input file path"
    )
    parser.add_argument(
        "-o", "--output", type=str, required=False, help="Output file path"
    )
    parser.add_argument(
        "-folder", "--folder", type=str, required=False, help="Input folder"
    )
    parser.add_argument(
        "-video", "--video", type=str, required=False, help="Input video"
    )
    parser.add_argument(
        "-z",
        "--zoom",
        type=float,
        required=False,
        default=1.25,
        help="Zoom level (default=1.25)",
    )
    parser.add_argument(
        "-s",
        "--shift",
        type=float,
        required=False,
        default=100,
        help="Shift level (default=100)",
    )
    parser.add_argument(
        "-start",
        "--start",
        type=float,
        required=False,
        default=0,
        help="Point of animation start (default=0.0)",
    )
    parser.add_argument(
        "-stop",
        "--stop",
        type=float,
        required=False,
        default=1.0,
        help="Point of animation stop (default=1.0)",
    )
    parser.add_argument(
        "-time",
        "--time",
        type=float,
        required=False,
        default=3.0,
        help="Time (in secs) for transition (total will be double)",
    )
    parser.add_argument(
        "-width",
        "--width",
        type=float,
        required=False,
        default=None,
        help="Zoom target pixel width (default is middle)",
    )
    parser.add_argument(
        "-height",
        "--height",
        type=float,
        required=False,
        default=None,
        help="Zoom target pixel height (default is middle)",
    )
    parser.add_argument(
        "-audio",
        "--audio",
        type=str,
        required=False,
        default=None,
        help="Audio file path",
    )
    parser.add_argument(
        "-face_zoom",
        "--face_zoom",
        required=False,
        default=False,
        action="store_true",
        help="USe detected faces as travel reference?",
    )
    parser.add_argument(
        "-reverse",
        "--reverse",
        required=False,
        default=False,
        action="store_true",
        help="Include reverse motion?",
    )
    parser.add_argument(
        "-step_factor",
        "--step_factor",
        required=False,
        default=False,
        action="store_true",
        help="step_factor increases the amount of zoom with each frame",
    )
    parser.add_argument(
        "-random_order",
        "--random_order",
        required=False,
        default=False,
        action="store_true",
        help="Randomise the order of images?",
    )
    parser.add_argument(
        "-random_zoom",
        "--random_zoom",
        required=False,
        default=False,
        action="store_true",
        help="Randomise the amount of zoom",
    )
    parser.add_argument(
        "-jitter",
        "--jitter",
        required=False,
        default=False,
        action="store_true",
        help="Randomise the amount of zoom",
    )
    parser.add_argument(
        "-duration",
        "--duration",
        required=False,
        default=None,
        help="Duration of the output clip",
    )

    args = parser.parse_args()

    if verbose:
        print_line(color=Color.WHITE)
        formatted_print("⚙️ Project options", bold=True, color=Color.CYAN)
        pprint.pprint(args)
        print_line(color=Color.WHITE)

    return args


def get_images(args) -> List[str]:
    if not args.folder:
        image_list = [args.input]
    else:
        image_list: List[str] = [
            str(img) for img in pathlib.Path(args.folder).glob("**/*")
        ]
    formatted_print(
        f"👡 Total of {len(image_list)} image(s)", bold=True, color=Color.MAGENTA
    )
    return sorted(image_list)


def validate_file_list(file_list: List[str]) -> None:
    for file in file_list:
        validate_file(file, verbose=False)
    print("✅ All image files valid")


def validate_file(file: str, verbose: bool = True) -> None:
    if not os.path.isfile(file):
        print(f"🔴 Invalid file: {file}")
        exit()
    if verbose:
        print(f"✅ Valid file: {file}")


def validate_all_input(image_paths: List[str], audio_file: str):
    validate_file_list(image_paths)
    if audio_file:
        validate_file(file=audio_file)


def get_time_list_from_audio_beats(audio_file: str) -> List[float]:
    if audio_file is None:
        return []
    y, sr = librosa.load(audio_file)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beat_times: np.ndarray = librosa.frames_to_time(beats, sr=sr)
    time_list = np.diff(beat_times)
    np.append(time_list, [1])
    return time_list.tolist()


def add_border_to_all_frames(frames: List[np.ndarray]) -> List[np.ndarray]:
    print(f"🎞 Frames #   : {len(frames)}")
    max_height = max(frame.shape[0] for frame in frames)
    print(f"📐 Max height : {max_height} pixels")
    max_width = max(frame.shape[1] for frame in frames)
    print(f"📐 Max width  : {max_width:4d} pixels")
    # Add border to make all images the same size
    print("🖼 Making all images same size ... ", end="")
    new_frames = []
    for frame in frames:
        frame_height, frame_width = frame.shape[0], frame.shape[1]
        top, bottom = split_int_in_two(
            value=int(max_height * (1 + DEFAULT_BORDER)) - frame_height
        )
        left, right = split_int_in_two(
            value=int(max_width * (1 + DEFAULT_BORDER)) - frame_width
        )
        bordered_frame = cv2.copyMakeBorder(
            src=frame,
            top=top,
            bottom=bottom,
            left=left,
            right=right,
            borderType=cv2.BORDER_REFLECT,
            value=[0, 0, 0],
        )
        new_frames.append(bordered_frame)
    print_success()
    return new_frames


def get_frame_time_map(fps: int) -> List[float]:
    pass


def move_image(image: np.ndarray, target_coords: Tuple[int, int]):
    pass


def build_images(args: argparse.Namespace) -> List[ProjectImage]:
    """

    """
    images = []
    for image_path in image_paths:
        images.append(ProjectImage(image_path=image_path))

    beats_list: List[float] = get_time_list_from_audio_beats(audio_file=args.audio)

    for image_idx, image in enumerate(images):
        # if using audio, sync time of each frame to the next beat, else use `args.time` throughout.
        if args.audio:
            image.time = beats_list[image_idx]
        else:
            image.time = args.time
        # randomise zoom levels
        if args.random_zoom:
            image.zoom = random.uniform(-2.0, 3.0)
        else:
            image.zoom = args.zoom
        image.reverse_ratio = 0.2

    if args.random_order:
        random.shuffle(images)
    return images


def create_video(
    args: argparse.Namespace, images: List[ProjectImage]
) -> ImageSequenceClip:
    """

    """
    all_frames = []
    for image in tqdm(images, desc="Processing images"):
        npyImage = cv2.imread(filename=image.image_path, flags=cv2.IMREAD_COLOR)

        intWidth = npyImage.shape[1]
        intHeight = npyImage.shape[0]

        fltRatio = float(intWidth) / float(intHeight)

        intWidth = min(int(1024 * fltRatio), 1024)
        intHeight = min(int(1024 / fltRatio), 1024)

        npyImage = cv2.resize(
            src=npyImage,
            dsize=(intWidth, intHeight),
            fx=0.0,
            fy=0.0,
            interpolation=cv2.INTER_AREA,
        )

        process_load(npyImage, {})

        objFrom = {
            "fltCenterU": args.width if args.width is not None else intWidth / 2.0,
            "fltCenterV": args.height if args.height is not None else intHeight / 2.0,
            "intCropWidth": int(math.floor(0.97 * intWidth)),
            "intCropHeight": int(math.floor(0.97 * intHeight)),
        }

        objTo = process_autozoom(
            objSettings={
                "fltShift": args.shift,
                "fltZoom": image.zoom,
                "objFrom": objFrom,
            }
        )

        npyResult = process_kenburns(
            objSettings={
                # num defines the number of discrete steps for the inwards transition
                "fltSteps": numpy.linspace(
                    start=args.start, stop=args.stop, num=int(image.time * FPS)
                ).tolist(),
                "objFrom": objFrom,
                "objTo": objTo,
                "boolInpaint": True,
            }
        )

        # Add reversal ratio
        breakpoint()
        image.frames = list(npyResult)

        if args.reverse:
            frame_list = npyResult + list(reversed(npyResult))[1:]
        else:
            frame_list = npyResult

        # Append to full list
        all_frames.extend(frame_list)

    # Split tensor lists in a list of frames
    frames_list: List[np.ndarray] = [npyFrame[:, :, ::-1] for npyFrame in all_frames]

    bordered_frames = add_border_to_all_frames(frames=frames_list)

    # Create output video
    video = moviepy.editor.ImageSequenceClip(sequence=bordered_frames, fps=FPS)
    return video


##########################################################
def run():
    args = parse_args()
    image_paths: List[str] = get_images(args)
    validate_all_input(image_paths=image_paths, audio_file=args.audio)

    images = build_images(args)
    image_chunks: List[List[ProjectImage]] = list(chunked(images, N_IMAGES_PER_CHUNK))

    videos: List[ImageSequenceClip] = []
    for image_chunk in tqdm(image_chunks, desc="Processing chunks"):
        videos.append(create_video(args=args, images=image_chunk))

    final_video = concatenate_videoclips(videos)
    if args.audio:
        print(f"🔊 Using audio from {args.audio}")
        final_video.set_audio(args.audio)
    final_video.write_videofile(filename=args.output, audio=args.audio)


##########################################################

if __name__ == "__main__":
    run()
