from enum import Enum


class EvaluationType(Enum):
    DEFAULT = 0
    CORRECTNESS = 1
    DETAILED_ORIENTATION = 2
    CONTEXT = 3
    TEMPORAL = 4
