"""
visualize.py — Visualization Module
=====================================
Builds the composite output image:
  - Left half : Image A (reference, untouched)
  - Right half : Image B (aligned) with:
      * Semi-transparent red fill over each flagged contour region
      * Solid red bounding box around each flagged region
      * Index label ("1", "2", …) at top-left of each bounding box

All visual parameters (alpha, colour, thickness) are loaded from config.json.

Public API
----------
build_composite(img_a, warped_b, contours, cfg, low_confidence=False)
    -> composite : np.ndarray (BGR)
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resize_to_same_height(img_a: np.ndarray, img_b: np.ndarray):
    """If heights differ, resize img_b to match img_a's height (keep aspect ratio)."""
    if img_a.shape[0] == img_b.shape[0]:
        return img_a, img_b
    h_a = img_a.shape[0]
    scale = h_a / img_b.shape[0]
    w_b_new = int(img_b.shape[1] * scale)
    img_b_resized = cv2.resize(img_b, (w_b_new, h_a), interpolation=cv2.INTER_LINEAR)
    return img_a, img_b_resized


def _draw_region(img: np.ndarray, contour: np.ndarray, index: int, cfg: dict, label: str = None):
    """
    Draw a bounding box and semitransparent fill for a flagged region.
    """
    x, y, w, h = cv2.boundingRect(contour)
    color = tuple(cfg["visualization"]["box_color"])
    thickness = cfg["visualization"]["box_thickness"]
    alpha = cfg["visualization"]["overlay_alpha"]
    font_scale = cfg["visualization"]["font_scale"]

    # Draw solid bounding box
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)

    # Semi-transparent fill over the exact contour
    overlay = img.copy()
    cv2.drawContours(overlay, [contour], -1, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # Draw label and index badge
    text = f"{index}"
    if label:
        text = f"{index}: {label}"
        
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    # Background rectangle for text
    cv2.rectangle(img, (x, max(0, y - th - 6)), (x + tw + 4, y), color, -1)
    cv2.putText(
        img,
        text,
        (x + 2, max(0, y - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _low_confidence_banner(canvas: np.ndarray):
    """Draw a visible warning banner at the top of the canvas."""
    h, w = canvas.shape[:2]
    banner_h = 32
    # Dark orange background bar
    cv2.rectangle(canvas, (0, 0), (w, banner_h), (0, 100, 200), -1)
    cv2.putText(
        canvas,
        "⚠  LOW CONFIDENCE ALIGNMENT — results may be inaccurate",
        (8, banner_h - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_composite(
    img_a: np.ndarray,
    warped_b: np.ndarray,
    contours: list,
    cfg: dict,
    low_confidence: bool = False,
    pct_area: float = 0.0,
    labels: list = None,
) -> np.ndarray:
    """
    Build the final side-by-side composite image.

    Parameters
    ----------
    img_a          : np.ndarray  Reference image A (BGR).
    warped_b       : np.ndarray  Aligned image B (BGR, same size as img_a after warp).
    contours       : list        Contours of flagged difference regions (from diff.py).
    cfg            : dict        Parsed config.json.
    low_confidence : bool        If True, draw a warning banner across the composite.

    Returns
    -------
    composite : np.ndarray  BGR composite image (width = 2 × max(w_a, w_b)).
    """
    # --- Ensure BGR ---
    if img_a.ndim == 2:
        img_a = cv2.cvtColor(img_a, cv2.COLOR_GRAY2BGR)
    if warped_b.ndim == 2:
        warped_b = cv2.cvtColor(warped_b, cv2.COLOR_GRAY2BGR)

    img_a, warped_b = _resize_to_same_height(img_a, warped_b)

    # Right panel: copy of warped_b onto which we draw overlays
    right_panel = warped_b.copy()

    # Draw each flagged region only if alignment is trustworthy
    if not low_confidence:
        for i, contour in enumerate(contours):
            label = labels[i] if labels and i < len(labels) else None
            _draw_region(right_panel, contour, i + 1, cfg, label)

    # Separator line between left and right panels
    sep = np.full((img_a.shape[0], 4, 3), 40, dtype=np.uint8)   # dark grey stripe

    composite = np.concatenate([img_a, sep, right_panel], axis=1)

    # Optional low-confidence banner
    if low_confidence:
        _low_confidence_banner(composite)

    # Column labels ("Reference" / "Inspection") below or above panels
    label_bar_h = 28
    label_bar = np.zeros((label_bar_h, composite.shape[1], 3), dtype=np.uint8)
    mid_left = img_a.shape[1] // 2
    mid_right = img_a.shape[1] + 4 + warped_b.shape[1] // 2

    def _center_text(bar, text, cx, color=(200, 200, 200)):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        x0 = max(0, cx - tw // 2)
        cv2.putText(bar, text, (x0, th + 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 1, cv2.LINE_AA)

    _center_text(label_bar, "REFERENCE (A)", mid_left)
    _center_text(label_bar, "INSPECTION (B)", mid_right, color=(100, 100, 255))

    n_diff = len(contours)
    if low_confidence:
        status_text = "Alignment Rejected (Differences Ignored)"
        status_color = (150, 150, 150)
    else:
        status_text = f"Differences detected: {n_diff} ({pct_area:.1f}% of frame)"
        status_color = (0, 100, 255) if n_diff > 0 else (0, 200, 80)
        
    _center_text(label_bar, status_text, composite.shape[1] - 160, color=status_color)

    composite = np.concatenate([label_bar, composite], axis=0)

    logger.info(
        "Composite built: size=%dx%d, differences=%d, low_confidence=%s",
        composite.shape[1], composite.shape[0], n_diff, low_confidence
    )
    return composite


def resize_for_display(composite: np.ndarray, max_width: int) -> np.ndarray:
    """
    Proportionally resize composite to fit within max_width for GUI display.
    """
    h, w = composite.shape[:2]
    if w <= max_width:
        return composite
    scale = max_width / w
    new_w = max_width
    new_h = int(h * scale)
    return cv2.resize(composite, (new_w, new_h), interpolation=cv2.INTER_AREA)
