from datetime import timedelta
from typing import List


def sum_timedeltas(timedeltas: List[timedelta]) -> timedelta:
    return sum(timedeltas, timedelta())
