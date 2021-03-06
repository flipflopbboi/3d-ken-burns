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
from tqdm import tqdm_notebook as tqdm

from audio import cluster_audio
from config import FPS, DEFAULT_BORDER, N_IMAGES_PER_CHUNK
from helpers.logging import print_line, formatted_print, Color, print_success
from helpers.numeric import split_int
from image import ProjectImage, ProjectFrame
from project import Project

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
        top, bottom = split_int(
            value=int(max_height * (1 + DEFAULT_BORDER)) - frame_height
        )
        left, right = split_int(
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


def build_images(
    args: argparse.Namespace, image_paths: List[str]
) -> List[ProjectImage]:
    """

    """
    images = []
    for image_path in image_paths:
        images.append(ProjectImage.from_args(image_path=image_path, args=args))

    beats_list: List[float] = get_time_list_from_audio_beats(audio_file=args.audio)
    audio_time_clusters = cluster_audio(audio_file=args.audio)

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
        image.set_array()
        image.resize(pixels=1024)

        process_load(npyImage=image.array, objSettings={})

        objFrom = {
            "fltCenterU": image.width / 2.0,
            "fltCenterV": image.height / 2.0,
            "intCropWidth": int(math.floor(0.97 * image.width)),
            "intCropHeight": int(math.floor(0.97 * image.height)),
        }

        objTo = process_autozoom(
            objSettings={
                "fltShift": image.shift,
                "fltZoom": image.zoom,
                "objFrom": objFrom,
            }
        )

        n_frames = int(image.time * FPS)
        npyResult: List[np.ndarray] = process_kenburns(
            objSettings={
                # num defines the number of discrete steps for the inwards transition
                "fltSteps": numpy.linspace(
                    start=image.start, stop=image.stop, num=n_frames
                ).tolist(),
                "objFrom": objFrom,
                "objTo": objTo,
                "boolInpaint": True,
            }
        )

        for frame_idx, frame in npyResult:
            image.frames.append(
                ProjectFrame(
                    idx=frame_idx,
                    image=image,
                    array=frame,
                    time=1 / FPS,
                    start=image.start + (1 / FPS),
                )
            )

        if args.reverse:
            frame_list = npyResult + list(reversed(npyResult))[1:]
        else:
            frame_list = npyResult

        # Append to full list
        all_frames.extend(frame_list)

    # Epic flash
    for image in images:
        image.frames[0].set_contrast_and_brightness()

    all_frames: List[np.ndarray] = [
        frame.array for image in images for frame in image.frames
    ]

    # Split tensor lists in a list of frames
    frames_list: List[np.ndarray] = [npyFrame[:, :, ::-1] for npyFrame in all_frames]

    bordered_frames = add_border_to_all_frames(frames=frames_list)

    # Create output video
    video = moviepy.editor.ImageSequenceClip(sequence=bordered_frames, fps=FPS)
    return video


##########################################################
def run():
    project = Project(args=parse_args())

    images = build_images(project.args, image_paths=project.image_paths)
    image_chunks: List[List[ProjectImage]] = list(chunked(images, N_IMAGES_PER_CHUNK))

    videos: List[ImageSequenceClip] = []
    for image_chunk in tqdm(image_chunks, desc="Processing chunks"):
        videos.append(create_video(args=project.args, images=image_chunk))

    final_video = concatenate_videoclips(videos)
    if project.args.audio:
        print(f"🔊 Using audio from {project.args.audio}")
        final_video.set_audio(project.args.audio)
    final_video.write_videofile(filename=project.args.output, audio=project.args.audio)


##########################################################

if __name__ == "__main__":
    run()
