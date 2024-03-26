"""
 Copyright (c) 2024, Deep Representation Learning Research Group, Seoul National University.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from enum import Enum


class EvaluationType(Enum):
    DEFAULT = 0
    CORRECTNESS = 1
    DETAILED_ORIENTATION = 2
    CONTEXT = 3
    TEMPORAL = 4
