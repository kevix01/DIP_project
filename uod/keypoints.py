"""Parsing of UrbanOmniDetect pose-model outputs into structured detections.

The model emits, per object, a 2D box and eight ordered keypoints that are the
projected corners of the object's 3D cuboid. Four of those eight are the
*ground-contact* corners that the BEV head consumes.

Ground-corner convention
------------------------
The paper text states indices 4-7 are ground; however the *released*
``urbanomnidetect_*`` checkpoints emit keypoints with the opposite ordering
(indices 0-3 are the ground corners, verified empirically: they sit ~25 px
below their vertical partners on the released models). To be correct for any
checkpoint regardless of its baked-in ordering, :class:`GroundIndexResolver`
auto-detects which half is the ground by majority vote over the first frames
(the ground corners are lower in the image, i.e. larger ``y``, than their
matching top corners) and then *locks* the choice for temporal stability. The
choice can also be forced explicitly.

Class unification
-----------------
Only two unified classes exist: 0=pedestrian, 2=vehicle.
All vehicle COCO classes (1,2,3,5,7) map to vehicle (2).
The original COCO class is preserved in `source_cls`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

__all__ = ["Detection", "GroundIndexResolver"]


@dataclass
class Detection:
    """A single detected object in full-image pixel coordinates.

    cls: unified class (0=pedestrian, 2=vehicle)
    source_cls: original COCO class (preserved for debugging/analysis)
    """
    cls: int
    conf: float
    xyxy: np.ndarray                  # (4,) [x1, y1, x2, y2]
    kpts: np.ndarray = field(        # (K, 2) all keypoints, may be empty
        default_factory=lambda: np.zeros((0, 2)))
    ground: np.ndarray = field(      # (4, 2) ground footprint, cyclic order
        default_factory=lambda: np.zeros((0, 2)))
    track_id: int = -1
    source_cls: Optional[int] = None  # original COCO class before unification
    geom_ok: bool = False             # trusted for rigid BEV footprint; raw camera points remain drawable


class GroundIndexResolver:
    """Decide which 4 of the 8 keypoints are the ground-contact corners.

    Parameters
    ----------
    forced : sequence of int, optional
        If given, always use exactly these indices (e.g. ``[0, 1, 2, 3]`` or
        ``[4, 5, 6, 7]``) and skip auto-detection.
    vote_frames : int
        Number of frames worth of detections to accumulate before locking the
        auto-detected choice.
    """

    def __init__(self, forced: Optional[Sequence[int]] = None,
                 vote_frames: int = 5):
        self._forced = list(forced) if forced is not None else None
        self.vote_frames = int(vote_frames)
        self._votes_lower_half = 0      # votes that the *second* half is ground
        self._votes_total = 0
        self._frames_seen = 0
        self._locked: Optional[List[int]] = list(forced) if forced else None

    @property
    def locked(self) -> bool:
        return self._locked is not None

    def indices(self, n_kpts: int) -> Optional[List[int]]:
        """Return the resolved ground indices, or ``None`` if not yet known."""
        if self._locked is not None:
            return self._locked
        if n_kpts >= 8:
            return None  # undecided
        # Degenerate model with <8 keypoints: treat all as "ground".
        return list(range(n_kpts))

    def observe(self, kpts_batch: np.ndarray) -> None:
        """Accumulate evidence from one frame's keypoints ``(N, K, 2)``."""
        if self._locked is not None or kpts_batch is None or len(kpts_batch) == 0:
            return
        k = kpts_batch.shape[1]
        if k == 0:
            return  # degenerate frame; do not lock to an empty index set
        if k < 8:
            self._locked = list(range(k))
            return
        half = k // 2
        # Per-detection: is the second half (indices half:) lower (larger y)?
        y_first = kpts_batch[:, :half, 1].mean(axis=1)
        y_second = kpts_batch[:, half:, 1].mean(axis=1)
        self._votes_lower_half += int(np.sum(y_second > y_first))
        self._votes_total += len(kpts_batch)
        self._frames_seen += 1
        if self._frames_seen >= self.vote_frames and self._votes_total > 0:
            second_is_ground = self._votes_lower_half > self._votes_total / 2
            self._locked = (list(range(half, k)) if second_is_ground
                            else list(range(half)))
