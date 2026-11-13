#!/usr/bin/env python3
"""Shared BEV building blocks for the v2 pipeline.

This module holds the pieces the live pipeline and the debug tool both need:

    pose model  -> detections (boxes + 8 keypoints)
    tracker     -> stable, jitter-damped, persistent tracks
    solver      -> orthogonality homography (warm-started, ~1-4 ms)
    viewport    -> temporally stable radar-style BEV
    renderer    -> source + BEV side by side with a timing HUD

It is a library, not a runnable script: the CLI lives in
:mod:`bev_realtime_v2` and the violation/TTC flow lives in
:mod:`debug_violations`.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from homography_rt import apply_homography
from uod import style
from uod.bev import BEVViewport
from uod.keypoints import Detection, GroundIndexResolver
from uod.model import load_detector

__all__ = ["BEVViewport", "Detection", "GroundIndexResolver", "RealtimePipeline",
           "RenderState", "Stopwatch", "build_argparser", "load_detector"]


@dataclass
class _TrackView:
    """An immutable snapshot of a track, safe to hand to the render thread."""
    id: int
    cls: int
    kpts: np.ndarray
    ground: np.ndarray
    box_xyxy: np.ndarray
    time_since_update: int
    bev_quad: object = None   # rigid footprint in rectified coords (or None)


@dataclass
class RenderState:
    """Everything the renderer needs for one frame, fully decoupled from the
    live tracker/solver state (so rendering can run on its own thread)."""
    frame: np.ndarray
    tracks: List[_TrackView]
    aux_dets: list
    aux_centers: np.ndarray
    H: np.ndarray
    info: object
    gi: Optional[list]
    img_wh: tuple
    timing: dict
    frame_idx: int
    n_gated: int
    bev_unit: float = 0.0   # median footprint size in H's gauge (BEV scale anchor)
    class_dims: dict = field(default_factory=dict)  # cls -> (w, l) in object_px units
    fps_core: float = 0.0


class Stopwatch:
    """Accumulates per-stage timings and exposes EMA-smoothed values (ms)."""

    def __init__(self, ema: float = 0.9):
        self.ema = ema
        self.t: dict = {}
        self._mark = None

    def start(self):
        self._mark = time.perf_counter()

    def lap(self, name: str):
        now = time.perf_counter()
        dt = (now - self._mark) * 1000.0
        self._mark = now
        self.t[name] = self.ema * self.t.get(name, dt) + (1 - self.ema) * dt
        return dt


class RealtimePipeline:
    """Base pipeline: owns the shared state used by the v2 tracker."""

    def __init__(self, pose_model, *, kp_imgsz=640,
                 kp_conf=0.25, device="cpu",
                 ground_indices=None, use_homography=True, show_3d=True,
                 snap_rect=True, trails=True, trail_len=16, stabilize=True,
                 bev_window=512, show_fov=True, tracker_kwargs=None,
                 bev_canvas=None, predict_fn=None):
        self.pose_model = pose_model
        self.aux_model = None
        self.draw_aux_box = True
        self.kp_imgsz = kp_imgsz
        self.kp_conf = kp_conf
        self.device = device
        self.use_homography = use_homography
        self.show_3d = show_3d
        self.snap_rect = snap_rect
        self.trails = trails
        self.trail_len = int(trail_len)
        self.show_fov = bool(show_fov)
        self._trails: dict = {}     # track id -> deque of BEV-canvas centres
        self.stabilize = bool(stabilize)
        # Footprint accumulation buffer: the ground plane is a scene constant,
        # so solving the homography over a rolling window of recent footprints
        # keeps it (and thus the BEV axis) stable when objects enter or leave.
        self._quad_buffer: deque = deque(maxlen=int(bev_window))
        self._cls_buffer: deque = deque(maxlen=int(bev_window))
        # The BEV scale (object_px / bev_unit) must be a CONSTANT, not a per-frame
        # buffer median -- otherwise the median pulses as footprints enter/leave
        # the window and every object slides radially (physically impossible).
        self._bev_unit_locked = None
        self._predict_fn = predict_fn  # override for sliced inference

        self.resolver = GroundIndexResolver(forced=ground_indices)
        self.viewport: Optional[BEVViewport] = None
        self._bev_canvas = bev_canvas
        self.sw = Stopwatch()
        self.frame_idx = 0
        self.fps_core = 0.0      # core (detect/track/solve) throughput, set by runner
        self.fps_render = 0.0    # render throughput, set by runner

    # ------------------------------------------------------------------ #
    @staticmethod
    def _class_footprint_dims(quads, classes, H):
        """Per-class median footprint (width, length) and the overall median size.

        A rigid object's footprint is a fixed real size, and objects of the same
        class are the same size, so we render every object of a class with one
        canonical box. We estimate that box as the robust per-class median of the
        footprint dimensions over the buffer (which spans many positions and
        frames), expressed relative to the overall median ``bev_unit``. The ratio
        is gauge-invariant and stable, so a class's rendered size is constant
        regardless of camera motion, an object's range, or scene turnover.
        Returns ``({cls: (w_ratio, l_ratio)}, bev_unit)``.
        """
        if not quads:
            return {}, 0.0
        pts = apply_homography(np.vstack(quads), H).reshape(len(quads), 4, 2)
        side_a = 0.5 * (np.linalg.norm(pts[:, 1] - pts[:, 0], axis=1) +
                        np.linalg.norm(pts[:, 3] - pts[:, 2], axis=1))
        side_b = 0.5 * (np.linalg.norm(pts[:, 2] - pts[:, 1], axis=1) +
                        np.linalg.norm(pts[:, 0] - pts[:, 3], axis=1))
        w = np.minimum(side_a, side_b)
        l = np.maximum(side_a, side_b)
        sz = np.sqrt(np.maximum(w * l, 0.0))
        ok = np.isfinite(sz) & (sz > 1e-6)
        if not ok.any():
            return {}, 0.0
        overall = float(np.median(sz[ok]))
        cls = np.asarray(classes)
        dims = {}
        for c in np.unique(cls[ok]):
            m = ok & (cls == c)
            dims[int(c)] = (float(np.median(w[m])) / overall,
                            float(np.median(l[m])) / overall)
        return dims, overall

    # ------------------------------------------------------------------ #
    def render(self, state: RenderState) -> np.ndarray:
        """Render one :class:`RenderState` into the composite (source | BEV).

        This is the expensive, quality-heavy path; it owns the BEV viewport and
        the motion-trail history, so it can run on a separate thread without
        touching the core's tracker state.
        """
        if self.viewport is None:
            cw = self._bev_canvas or state.img_wh
            self.viewport = BEVViewport(canvas_size=cw)
        if self.use_homography:
            self.viewport.update(state.H, state.img_wh, unit=state.bev_unit)
        return self._render(state)

    # ------------------------------------------------------------------ #
    def _draw_trails(self, bev, cls_of):
        """Draw fading per-track motion trails directly on the BEV canvas.

        Fade is faked by lerping the class colour toward the background, which
        avoids a per-segment alpha blend and keeps trails under the footprints.
        """
        bg = np.array(style.THEME["bev_bg"], dtype=np.float32)
        for tid, dq in self._trails.items():
            pj = list(dq)
            if len(pj) < 3:
                continue
            base = np.array(style.instance_color(cls_of.get(tid, -1), tid, "pose"),
                            dtype=np.float32)
            n = len(pj)
            for i in range(n - 1):
                f = i / (n - 1)
                col = bg * (1.0 - (0.12 + 0.5 * f)) + base * (0.12 + 0.5 * f)
                cv2.line(bev, (int(pj[i][0]), int(pj[i][1])),
                         (int(pj[i + 1][0]), int(pj[i + 1][1])),
                         (int(col[2]), int(col[1]), int(col[0])),
                         2, cv2.LINE_AA)


# --------------------------------------------------------------------------- #
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                allow_abbrev=False)
    p.add_argument("--input", required=True, help="video path, image dir, or webcam index")
    p.add_argument("--kp-model", required=True, help="UrbanOmniDetect pose checkpoint")
    p.add_argument("--kp-imgsz", type=int, default=640)
    p.add_argument("--kp-conf", type=float, default=0.25)
    p.add_argument("--homography", choices=["ortho", "adam", "none"], default="ortho",
                   help="BEV solver ('adam' is an alias for the fast LM "
                        "orthogonality solver kept for CLI compatibility)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--export", default="none",
                   choices=["none", "engine", "tensorrt", "onnx"])
    p.add_argument("--half", action="store_true", help="FP16 for the exported engine")
    p.add_argument("--ground-indices", default=None,
                   help="force ground keypoint indices, e.g. '0,1,2,3'")
    p.add_argument("--no-3d", action="store_true", help="draw only ground quads")
    p.add_argument("--no-snap-rect", action="store_true",
                   help="show raw (unsnapped) BEV footprints")
    p.add_argument("--no-trails", action="store_true",
                   help="disable fading BEV motion trails")
    p.add_argument("--trail-len", type=int, default=16,
                   help="number of frames of BEV motion trail to keep")
    p.add_argument("--no-stabilize", action="store_true",
                   help="solve the BEV homography per frame instead of over a "
                        "rolling footprint buffer (less stable axis)")
    p.add_argument("--bev-window", type=int, default=512,
                   help="rolling buffer size (footprints) for a stable BEV axis; "
                        "larger = steadier scale but more lag on camera motion")
    p.add_argument("--max-age", type=int, default=30)
    p.add_argument("--min-hits", type=int, default=2)
    p.add_argument("--output", default=None, help="write annotated mp4 here")
    p.add_argument("--display", action="store_true", help="show a live window")
    p.add_argument("--max-frames", type=int, default=0, help="0 = all")
    p.add_argument("--smooth", type=int, default=0,
                   help="non-causal offline smoothing window in frames (0 = off, "
                        "live streaming). Buffers the whole clip, then zero-phase "
                        "filters every track's pose and the homography for a "
                        "jitter-free render. Odd windows recommended (e.g. 11).")
    p.add_argument("--no-fov", action="store_true",
                   help="do not draw the camera's visible field of view on the radar")
    return p


def _ema(prev, dt):
    f = 1.0 / dt if dt > 0 else 0.0
    return f if prev <= 0 else 0.9 * prev + 0.1 * f
