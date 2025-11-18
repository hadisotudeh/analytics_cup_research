from datetime import datetime
import pandas as pd
import numpy as np


def format_time(t_str: str) -> str:
    t = datetime.strptime(t_str, "%H:%M:%S")
    return f"{t.hour * 60 + t.minute:02}:{t.second:02}"


def substitution_condition(start_time):

    mmss_str = format_time(start_time)

    minutes, seconds = map(int, mmss_str.split(":"))

    if (minutes * 60 + seconds) > 80 * 60:  # 80 minutes
        return True
    return False
