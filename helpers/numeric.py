import math
from typing import Tuple


def split_int(value: int, ratio: float = 0.5) -> Tuple[int, int]:
    """

    """
    if type(value) is not int:
        raise TypeError(f"Expected int but got {type(value)}")

    return math.ceil(value * ratio), math.floor(value * (1 - ratio))
