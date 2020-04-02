import sys
import time
from typing import TypeVar, Optional

TAB = "     "

ColorType = TypeVar("ColorType", bound=str)


class Color:
    WHITE: ColorType = "\033[97m"
    PURPLE: ColorType = "\033[95m"
    CYAN: ColorType = "\033[96m"
    DARKCYAN: ColorType = "\033[36m"
    BLUE: ColorType = "\033[94m"
    GREEN: ColorType = "\033[92m"
    YELLOW: ColorType = "\033[93m"
    RED: ColorType = "\033[91m"
    GREY: ColorType = "\033[90m"
    MAGENTA: ColorType = "\033[95m"
    BOLD: ColorType = "\033[1m"
    UNDERLINE: ColorType = "\033[4m"
    END: ColorType = "\033[0m"


def formatted_print(
    msg: str,
    end: Optional[str] = None,
    indent_lvl: int = 0,
    color: Optional[ColorType] = None,
    bold: bool = False,
    underline: bool = False,
) -> None:
    prefix = ""
    prefix += TAB * indent_lvl
    if bold:
        prefix += Color.BOLD
    if underline:
        prefix += Color.UNDERLINE
    if color:
        prefix += color
    print(prefix + msg + Color.END, end=end)


def print_line(color: Optional[ColorType] = None) -> None:
    formatted_print("-" * 80, color=color)


def print_success() -> None:
    print("OK ✅")


def print_error(e: Exception) -> None:
    print(TAB + "ERROR ❌")
    print(TAB + f"💥 {type(e).__name__}")


def animated_wait(time_in_secs: float) -> None:
    FPS = 4
    animation = "|/-\\"
    for i in range(int(time_in_secs * FPS)):
        time.sleep(FPS / time_in_secs)
        sys.stdout.write("\r" + animation[i % len(animation)])
        sys.stdout.flush()
