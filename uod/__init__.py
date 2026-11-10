"""UrbanOmniDetect inference / real-time BEV runtime package.

Modules
-------
keypoints   parse pose-model outputs; resolve ground-corner indices
bev         temporally stable radar-style BEV viewport + rendering
bev_ocsort  real-time BEV tracker with motion-coasted persistence
viz         drawing helpers for the source frame and BEV
textdraw    HUD text layer
model       model loading + optional TensorRT export
classes     unified class mapping (0=pedestrian, 2=vehicle)

The orthogonality homography solver lives in the top-level module
``homography_rt`` (named to match the repository structure).
"""

__version__ = "0.1.0"
