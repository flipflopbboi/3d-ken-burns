import argparse
import os
import pathlib
from dataclasses import dataclass
from typing import List

from helpers.logging import formatted_print, Color


@dataclass
class Project:
    args: argparse.Namespace

    def __post_init__(self):
        self.image_paths: List[str] = self.get_images()
        self.validate()

    def get_images(self) -> List[str]:
        if not self.args.folder:
            image_list = [self.args.input]
        else:
            image_list: List[str] = [
                str(img) for img in pathlib.Path(self.args.folder).glob("**/*")
            ]
        formatted_print(
            f"👡 Total of {len(image_list)} image(s)", bold=True, color=Color.MAGENTA
        )
        return sorted(image_list)

    def validate(self):
        self.validate_file_list(self.image_paths)
        if self.args.audio:
            self.validate_file(file=self.args.audio)

    def validate_file(self, file: str, verbose: bool = True) -> None:
        if not os.path.isfile(file):
            print(f"🔴 Invalid file: {file}")
            exit()
        if verbose:
            print(f"✅ Valid file: {file}")

    def validate_file_list(self, file_list: List[str]) -> None:
        for file in file_list:
            self.validate_file(file, verbose=False)
        print("✅ All image files valid")
