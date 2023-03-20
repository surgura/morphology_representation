from typing import List
import argparse


def indices_type(arg: str) -> List[int]:
    if arg.isnumeric():
        return [int(arg)]
    else:
        parts = arg.split(":")
        if len(parts) != 2 or not parts[0].isnumeric() or not parts[1].isnumeric():
            raise argparse.ArgumentTypeError(
                "Indices type must be of the form <integer> (single run) or <integer>:<integer> (run range, excluding upper)."
            )
        low = int(parts[0])
        high = int(parts[1])
        if low > high:
            raise argparse.ArgumentTypeError(
                "Indices type must be of the form <integer> (single run) or <integer>:<integer> (run range, excluding upper)."
            )
        return [i for i in range(low, high)]
