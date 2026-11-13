"""Batched, anti-aliased UI rendering (text, chips, panels) via Pillow.

OpenCV's Hershey fonts are the single biggest "hobby project" tell. This module
collects every UI element for a frame (rounded panels, label chips, text, thin
lines) and composites them in **one** Pillow pass with proper TrueType fonts and
real alpha blending, then hands back a BGR array. One conversion per frame keeps
it real-time.

All colours are RGB (0-255), matching :mod:`uod.style`. If Pillow or the fonts
are unavailable, :class:`UILayer` falls back to OpenCV so rendering still works.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    _HAVE_PIL = True
except Exception:
    _HAVE_PIL = False

RGB = Tuple[int, int, int]

# Font search paths (Liberation Sans ~ Arial; Liberation Mono for numerals).
_FONT_CANDIDATES = {
    "regular": ["/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
    "bold":    ["/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
    "mono":    ["/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"],
}

__all__ = ["UILayer"]


def _first_existing(paths: List[str]) -> Optional[str]:
    for p in paths:
        if os.path.exists(p):
            return p
    return None


class UILayer:
    """Collects UI primitives and composites them once over a BGR image."""

    _font_cache: dict = {}
    _metrics_cache: dict = {}

    def __init__(self):
        self._panels = []   # (xy0, xy1, rgb, alpha, radius)
        self._chips = []    # dict
        self._texts = []    # dict
        self._lines = []    # (p0, p1, rgb, width, alpha)

    # -- collectors ---------------------------------------------------- #
    def panel(self, xy0, xy1, color: RGB, alpha: float = 0.55, radius: int = 10):
        self._panels.append((tuple(xy0), tuple(xy1), color, alpha, radius))

    def line(self, p0, p1, color: RGB, width: int = 1, alpha: float = 1.0):
        self._lines.append((tuple(p0), tuple(p1), color, width, alpha))

    def text(self, x, y, s, *, size=14, color: RGB = (240, 240, 240),
             font="regular", anchor="la", shadow=True, alpha=1.0):
        self._texts.append(dict(x=x, y=y, s=s, size=size, color=color,
                                font=font, anchor=anchor, shadow=shadow,
                                alpha=alpha))

    def chip(self, x, y, s, *, size=13, fg: RGB = (245, 245, 245),
             bg: RGB = (30, 30, 30), alpha: float = 0.92, pad=(6, 3),
             radius=7, anchor="la", font="bold", accent: Optional[RGB] = None):
        """A rounded label chip with text. ``anchor`` positions the chip box."""
        self._chips.append(dict(x=x, y=y, s=s, size=size, fg=fg, bg=bg,
                                alpha=alpha, pad=pad, radius=radius,
                                anchor=anchor, font=font, accent=accent))

    # -- fonts --------------------------------------------------------- #
    @classmethod
    def _font(cls, style: str, size: int):
        key = (style, int(size))
        if key in cls._font_cache:
            return cls._font_cache[key]
        f = None
        if _HAVE_PIL:
            path = _first_existing(_FONT_CANDIDATES.get(style, _FONT_CANDIDATES["regular"]))
            if path:
                try:
                    f = ImageFont.truetype(path, int(size))
                except Exception:
                    f = None
            if f is None:
                f = ImageFont.load_default()
        cls._font_cache[key] = f
        return f

    @staticmethod
    def _anchor_xy(x, y, w, h, anchor):
        """Top-left of a (w, h) box given an anchor code like 'la','ma','mm'."""
        ax = {"l": x, "m": x - w / 2, "r": x - w}[anchor[0]]
        ay = {"a": y, "m": y - h / 2, "d": y - h, "s": y - h}[anchor[1]]
        return ax, ay

    @classmethod
    def _chip_metrics(cls, font_name: str, size: int, s: str):
        """Text metrics (l, t, r, b) for a chip string, memoised.

        The same chip strings repeat every frame, so the glyph measurement is
        cached instead of being recomputed for each occurrence.
        """
        key = (font_name, int(size), s)
        hit = cls._metrics_cache.get(key)
        if hit is not None:
            return hit
        font = cls._font(font_name, size)
        try:
            l, t, r, b = font.getbbox(s)
        except Exception:
            l = t = 0
            r = b = 0
        hit = (l, t, r, b)
        if len(cls._metrics_cache) > 8192:
            cls._metrics_cache.clear()
        cls._metrics_cache[key] = hit
        return hit

    # -- render -------------------------------------------------------- #
    def _bounds(self, w: int, h: int):
        """Bounding box (x0, y0, x1, y1) covering every collected primitive.

        The overlay is allocated (and converted to numpy) only over this
        region instead of the whole frame: the UI touches a small fraction
        of the pixels, but a full-size RGBA buffer costs megabytes per frame.
        Extents are measured with the real font metrics so no primitive can
        ever spill outside the allocated region.
        """
        INF = float("inf")
        x0, y0, x1, y1 = INF, INF, -INF, -INF

        def acc(px, py):
            nonlocal x0, y0, x1, y1
            if px < x0: x0 = px
            if py < y0: y0 = py
            if px > x1: x1 = px
            if py > y1: y1 = py

        for (p0, p1, _rgb, width, _alpha) in self._lines:
            acc(p0[0] - width, p0[1] - width)
            acc(p0[0] + width, p0[1] + width)
            acc(p1[0] - width, p1[1] - width)
            acc(p1[0] + width, p1[1] + width)
        for (xy0, xy1, _rgb, _alpha, radius) in self._panels:
            r = radius + 2
            acc(xy0[0] - r, xy0[1] - r)
            acc(xy1[0] + r, xy1[1] + r)

        for c in self._chips:
            l, t, r, b = self._chip_metrics(c["font"], c["size"], c["s"])
            tw, th = r - l, b - t
            px, py = c["pad"]
            bw2, bh2 = tw + 2 * px, th + 2 * py
            cx, cy = self._anchor_xy(c["x"], c["y"], bw2, bh2, c["anchor"])
            cx, cy = int(cx), int(cy)
            acc(cx, cy)
            acc(cx + bw2, cy + bh2)
            acc(cx + px - l, cy + py - t)
            acc(cx + px - l + tw, cy + py - t + th)

        for tx in self._texts:
            font = self._font(tx["font"], tx["size"])
            size = int(tx["size"])
            try:
                l, t, r, b = ImageDraw.Draw(Image.new("L", (1, 1))).textbbox(
                    (0, 0), tx["s"], font=font, anchor=tx["anchor"])
                acc(tx["x"] + l, tx["y"] + t)
                acc(tx["x"] + r, tx["y"] + b)
            except Exception:
                acc(tx["x"] - 4 * size, tx["y"] - 2 * size)
                acc(tx["x"] + 4 * size, tx["y"] + 2 * size)
            if tx["shadow"]:
                acc(tx["x"] + 1, tx["y"] + 1)
            acc(tx["x"] - 4 * size, tx["y"] - 2 * size)
            acc(tx["x"] + 4 * size, tx["y"] + 2 * size)

        if x0 == INF:
            return None
        pad = 4
        x0 = max(0, int(x0) - pad)
        y0 = max(0, int(y0) - pad)
        x1 = min(w, int(x1) + pad)
        y1 = min(h, int(y1) + pad)
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1

    def render(self, bgr: np.ndarray) -> np.ndarray:
        if not (self._panels or self._chips or self._texts or self._lines):
            return bgr
        if not _HAVE_PIL:
            return self._render_cv(bgr)

        h, w = bgr.shape[:2]
        bounds = self._bounds(w, h)
        if bounds is None:
            return bgr
        bx0, by0, bx1, by1 = bounds
        bw, bh = bx1 - bx0, by1 - by0

        overlay = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)

        for (p0, p1, rgb, width, alpha) in self._lines:
            d.line([(int(p0[0]) - bx0, int(p0[1]) - by0),
                    (int(p1[0]) - bx0, int(p1[1]) - by0)],
                   fill=(rgb[0], rgb[1], rgb[2], int(255 * alpha)),
                   width=int(width))

        for (xy0, xy1, rgb, alpha, radius) in self._panels:
            box = [int(xy0[0]) - bx0, int(xy0[1]) - by0,
                   int(xy1[0]) - bx0, int(xy1[1]) - by0]
            d.rounded_rectangle(box, radius=radius,
                                fill=(rgb[0], rgb[1], rgb[2], int(255 * alpha)))

        for c in self._chips:
            font = self._font(c["font"], c["size"])
            l, t, r, b = self._chip_metrics(c["font"], c["size"], c["s"])
            tw, th = r - l, b - t
            px, py = c["pad"]
            bw2, bh2 = tw + 2 * px, th + 2 * py
            cx, cy = self._anchor_xy(c["x"], c["y"], bw2, bh2, c["anchor"])
            cx, cy = int(cx) - bx0, int(cy) - by0
            bg = c["bg"]
            d.rounded_rectangle([cx, cy, cx + bw2, cy + bh2], radius=c["radius"],
                                fill=(bg[0], bg[1], bg[2], int(255 * c["alpha"])))
            if c["accent"] is not None:
                a = c["accent"]
                d.rounded_rectangle([cx, cy, cx + 3, cy + bh2],
                                    radius=2, fill=(a[0], a[1], a[2], 255))
            fg = c["fg"]
            d.text((cx + px - l, cy + py - t), c["s"], font=font,
                   fill=(fg[0], fg[1], fg[2], 255))

        for tx in self._texts:
            font = self._font(tx["font"], tx["size"])
            col = tx["color"]
            a = int(255 * tx["alpha"])
            try:
                if tx["shadow"]:
                    d.text((tx["x"] + 1 - bx0, tx["y"] + 1 - by0), tx["s"], font=font,
                           fill=(0, 0, 0, int(a * 0.6)), anchor=tx["anchor"])
                d.text((tx["x"] - bx0, tx["y"] - by0), tx["s"], font=font,
                       fill=(col[0], col[1], col[2], a), anchor=tx["anchor"])
            except Exception:
                d.text((tx["x"] - bx0, tx["y"] - by0), tx["s"], font=font,
                       fill=(col[0], col[1], col[2], a))

        # Composite the overlay onto the BGR image. PIL only rasterised the
        # (cheap) text/shapes; the blend touches only the pixels the UI
        # actually covers (alpha > 0), which are a small fraction of the
        # frame. The touched pixels are addressed by their (row, col) index
        # rather than by a boolean mask: it avoids materialising a second
        # full-region float buffer and is bit-identical.
        ov = np.asarray(overlay)               # (bh, bw, 4) RGBA
        m = ov[:, :, 3] > 0
        if not m.any():
            return bgr
        region = bgr[by0:by1, bx0:bx1]
        idx = np.nonzero(m)
        a = ov[:, :, 3][idx].astype(np.float32).reshape(-1, 1) * (1.0 / 255.0)
        fg = ov[:, :, 2::-1][idx].astype(np.float32)   # RGB -> BGR
        roi = region[idx].astype(np.float32)
        region[idx] = (roi * (1.0 - a) + fg * a).astype(np.uint8)
        return bgr

    # -- OpenCV fallback ---------------------------------------------- #
    def _render_cv(self, bgr):
        import cv2
        for (p0, p1, rgb, width, alpha) in self._lines:
            cv2.line(bgr, tuple(map(int, p0)), tuple(map(int, p1)),
                     (rgb[2], rgb[1], rgb[0]), int(width), cv2.LINE_AA)
        for (xy0, xy1, rgb, alpha, radius) in self._panels:
            ov = bgr.copy()
            cv2.rectangle(ov, (int(xy0[0]), int(xy0[1])), (int(xy1[0]), int(xy1[1])),
                          (rgb[2], rgb[1], rgb[0]), -1)
            cv2.addWeighted(ov, alpha, bgr, 1 - alpha, 0, bgr)
        for c in self._chips + self._texts:
            s, x, y = c["s"], int(c["x"]), int(c["y"])
            col = c.get("fg", c.get("color", (240, 240, 240)))
            cv2.putText(bgr, s, (x, y + 12), cv2.FONT_HERSHEY_SIMPLEX,
                        c["size"] / 28.0, (col[2], col[1], col[0]), 1, cv2.LINE_AA)
        return bgr
