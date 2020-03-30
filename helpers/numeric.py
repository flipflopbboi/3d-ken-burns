from typing import Tuple


def split_int_in_half(value: int) -> Tuple[int, int]:
    if type(value) is not int:
        raise TypeError(f"Expected int but got {type(value)}")

    if value % 2 == 0:
        return value // 2, value // 2

    return value // 2, value // 2 + 1
