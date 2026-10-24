"""Cut a table photo into five fixed zones: the four players' hands and the center pile.

The photo is divided on a 20 x 20 grid. Each hand is rotated so the player's cards are
upright, so the network always sees a hand from its owner's side:

    p1 bottom   rows 15-20, cols 5-16    as is
    p2 right    rows 5-17,  cols 14-20   rotated 90 degrees clockwise
    p3 top      rows 0-6,   cols 5-15    rotated 180 degrees
    p4 left     rows 5-18,  cols 0-6     rotated 90 degrees counter-clockwise
    center      rows 8-14,  cols 8-12    as is

The zones overlap at the corners, so a card lying in a corner can be counted twice.
"""

from typing import List, NamedTuple

import numpy as np

ZONE_NAMES = ("p1", "p2", "p3", "p4", "center")


class Zone(NamedTuple):
    name: str
    x0: int
    y0: int
    x1: int
    y1: int
    rot90_k: int   # passed to np.rot90 (negative = clockwise)


def zone_layout(height: int, width: int) -> List[Zone]:
    """The five zones of a height x width photo, in ZONE_NAMES order, as pixel boxes."""
    h, w = height // 20, width // 20
    return [
        Zone("p1", 5 * w, 15 * h, 16 * w, height, 0),
        Zone("p2", 14 * w, 5 * h, width, 17 * h, -1),
        Zone("p3", 5 * w, 0, 15 * w, 6 * h, 2),
        Zone("p4", 0, 5 * h, 6 * w, 18 * h, 1),
        Zone("center", 8 * w, 8 * h, 12 * w, 14 * h, 0),
    ]


def crop_zones(image: np.ndarray) -> List[np.ndarray]:
    """An H x W x C photo -> the five upright crops, in ZONE_NAMES order."""
    crops = []
    for z in zone_layout(*image.shape[:2]):
        crop = image[z.y0:z.y1, z.x0:z.x1]
        crops.append(np.ascontiguousarray(np.rot90(crop, k=z.rot90_k)) if z.rot90_k else crop)
    return crops
