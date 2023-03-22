from typing import List, Optional
import argparse


def indices_type(all_indices: List[int]):
    def impl(arg: str) -> Optional[List[int]]:
        if arg.isnumeric():
            argint = int(arg)
            if argint not in all_indices:
                raise ValueError(
                    f"Provided index {argint} not in allowed range {all_indices}"
                )
            return [argint]
        elif arg == "all":
            return all_indices
        else:
            parts = arg.split(":")
            if len(parts) != 2 or not parts[0].isnumeric() or not parts[1].isnumeric():
                raise argparse.ArgumentTypeError(
                    "Indices type must be of the form 'all', <integer> (single run), or <integer>:<integer> (run range, excluding upper)."
                )
            low = int(parts[0])
            high = int(parts[1])
            if low > high:
                raise argparse.ArgumentTypeError(
                    "Indices type must be of the form 'all', <integer> (single run), or <integer>:<integer> (run range, excluding upper)."
                )
            argints = [i for i in range(low, high)]
            if not all([i in all_indices for i in argints]):
                raise ValueError(
                    f"Provided indices {argints} not in allowed range {all_indices}"
                )

            return argints

    return impl
