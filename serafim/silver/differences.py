"""Difference images between seasons (task D6).

Which differences exist is agreed with AI & validation and lives here, so the simulated gold
and the real pipeline make exactly the same ones (93 for the 5 targets).
"""

from __future__ import annotations

from collections.abc import Iterable


def difference_pairs(cells: Iterable[tuple[int, int]]) -> list[tuple[int, int, int]]:
    """(detector, season_a, season_b) of every difference to make, in that order.

    `cells` are the (detector, season) cells that exist; the white image (detector 0) and
    "all seasons" (season 0) are not cells. Per detector, with the seasons it has:

    - each season minus the one before it (n - 1 differences), and
    - if it has 3 or more seasons, also the last minus the first.

    The image is season_b minus season_a, and season_a < season_b. A detector with 2 seasons
    has one difference (last minus first is the same pair), and with 1 season none. A season
    the detector does not have is skipped: nothing is made up for a missing cell.
    """
    by_detector: dict[int, set[int]] = {}
    for detector, season in cells:
        if detector >= 1 and season >= 1:
            by_detector.setdefault(int(detector), set()).add(int(season))
    pairs: list[tuple[int, int, int]] = []
    for detector in sorted(by_detector):
        seasons = sorted(by_detector[detector])
        found = list(zip(seasons, seasons[1:], strict=False))
        if len(seasons) >= 3:
            found.append((seasons[0], seasons[-1]))
        pairs += [(detector, a, b) for a, b in found]
    return pairs
