#!/usr/bin/env python
import argparse
import torch
import torchvision

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

arguments_strIn = "./images/doublestrike.jpg"
arguments_strOut = "./autozoom.mp4"

FPS = 25

##########################################################


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", type=str, required=True, help="Input file path"
    )
    parser.add_argument(
        "-o", "--output", type=str, required=True, help="Output file path"
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
        type=int,
        required=False,
        default=None,
        help="Zoom target pixel width (default is middle)",
    )
    parser.add_argument(
        "-height",
        "--height",
        type=int,
        required=False,
        default=None,
        help="Zoom target pixel height (default is middle)",
    )
    parser.add_argument(
        "-reverse",
        "--reverse",
        required=False,
        default=False,
        action='store_true',
        help="Include reverse motion?",
    )
    return parser.parse_args()


##########################################################

if __name__ == "__main__":
    args = parse_args()
    npyImage = cv2.imread(filename=args.input, flags=cv2.IMREAD_COLOR)

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
        "fltCenterU": args.width if args.width else intWidth / 2.0,
        "fltCenterV": args.height if args.height else intHeight / 2.0,
        "intCropWidth": int(math.floor(0.97 * intWidth)),
        "intCropHeight": int(math.floor(0.97 * intHeight)),
    }

    objTo = process_autozoom(
        {"fltShift": args.shift, "fltZoom": args.zoom, "objFrom": objFrom}
    )

    npyResult = process_kenburns(
        {
            # num defines the number of discrete steps for the inwards transition
            "fltSteps": numpy.linspace(
                start=args.start, stop=args.stop, num=int(args.time * FPS)
            ).tolist(),
            "objFrom": objFrom,
            "objTo": objTo,
            "boolInpaint": True,
        }
    )

    # Create output video
    if args.reverse:
        frame_list = npyResult + list(reversed(npyResult))[1:]
    else:
        frame_list = npyResult
    moviepy.editor.ImageSequenceClip(
        sequence=[
            npyFrame[:, :, ::-1]
            for npyFrame in npyResult + frame_list
        ],
        fps=FPS,
    ).write_videofile(args.output)
