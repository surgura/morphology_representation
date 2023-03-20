from typing import List


def runs_type(arg: str) -> List[int]:
    if arg.isnumeric():
        return [int(arg)]
    else:
        parts = arg.split(":")
        if len(parts) != 2 or not parts[0].isnumeric() or not parts[1].isnumeric():
            raise ValueError()
        low = int(parts[0])
        high = int(parts[1])
        if low > high:
            raise ValueError()
        return [i for i in range(low, high + 1)]
