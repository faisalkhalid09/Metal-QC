"""
align.py -- Image Alignment Module
====================================
Registers Image B onto the coordinate frame of Image A using:
  1. ORB keypoint detection + BFMatcher + RANSAC homography (primary)
  2. Phase-correlation translation-only fallback (if too few matches or degenerate H)

Background isolation (ROI masking)
-----------------------------------
Shop photos often contain cluttered backgrounds (people, cables, equipment).
ORB will grab keypoints from whatever has the most texture, which may be
the background rather than the assembly. This causes misalignment.

When cfg["alignment"]["use_roi_mask"] is true, a coarse center-crop mask
is applied before feature detection. The mask keeps the central 70% (x) x
80% (y) of the frame where the assembly typically sits, blocking edge clutter.
This is conservative: it only excludes extreme borders, not arbitrary regions.
For severe clutter cases, a brightness-based mask is also optionally applied
(cfg["alignment"]["use_brightness_mask"]) to prefer lighter metal regions
over dark backgrounds.

Public API
----------
align_images(img_a, img_b, cfg) -> (warped_b, confidence, method_used, H)
test_alignment(img, cfg)        -> pixel_error (float)
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_gray(img: np.ndarray) -> np.ndarray:
    """Convert BGR or grayscale image to single-channel uint8."""
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.copy()


def _is_degenerate_homography(H: np.ndarray, img_shape: tuple, max_scale: float = 2.5) -> bool:
    """
    Return True if H looks degenerate:
      - Nan/Inf values
      - Negative determinant (axis flip / reflection)
      - Scale change beyond max_scale (extreme zoom or shrink)
      - Extreme shear (warps image into a very thin strip)
    max_scale tightened from 5.0 to 2.5 to catch bad alignments on real photos.
    """
    if H is None:
        return True
    if not np.isfinite(H).all():
        return True

    det = np.linalg.det(H[:2, :2])
    # Negative det = orientation flip (reflection) -- always degenerate for photos
    if det <= 0:
        logger.warning("Homography has negative determinant (%.4f) -- degenerate.", det)
        return True
    if abs(det) < 1e-6:
        return True

    h, w = img_shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners, H)
    area_original = w * h
    pts = warped_corners.reshape(4, 2)
    x, y = pts[:, 0], pts[:, 1]
    area_warped = 0.5 * abs(
        np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))
    )
    if area_original == 0:
        return True
    scale_ratio = area_warped / area_original
    if scale_ratio > max_scale or scale_ratio < 1.0 / max_scale:
        logger.warning(
            "Homography scale ratio %.2f exceeds threshold %.1f -- degenerate.",
            scale_ratio, max_scale
        )
        return True

    # Check for extreme shear: the warped image's minimum bounding dimension
    # should not be less than 10% of the original
    warped_w = float(np.linalg.norm(warped_corners[1] - warped_corners[0]))
    warped_h = float(np.linalg.norm(warped_corners[3] - warped_corners[0]))
    if warped_w < 0.10 * w or warped_h < 0.10 * h:
        logger.warning("Homography produces extreme shear -- degenerate.")
        return True

    return False


def _phase_correlation_align(img_a_gray: np.ndarray, img_b_gray: np.ndarray) -> np.ndarray:
    """
    Estimate pure translation between two images via phase correlation.
    Returns a 3x3 translation homography matrix.
    """
    fa = img_a_gray.astype(np.float32)
    fb = img_b_gray.astype(np.float32)
    (dx, dy), _ = cv2.phaseCorrelate(fa, fb)
    H = np.eye(3, dtype=np.float64)
    H[0, 2] = dx
    H[1, 2] = dy
    return H


# ---------------------------------------------------------------------------
# ROI masking for background isolation
# ---------------------------------------------------------------------------

def _build_roi_mask(gray: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Build a binary mask to restrict ORB feature detection to the assembly region.

    Strategy (applied in order, combined with bitwise AND):
      1. Center-crop mask: keeps the central x_frac x y_frac region.
         Discards extreme edges where background clutter accumulates.
      2. Optional brightness mask: keeps pixels above a percentile brightness
         threshold. Useful for bright stainless steel vs dark backgrounds.
         Disabled for dark-metal assemblies (steel posts, black brackets).

    Returns uint8 mask (255 = include, 0 = exclude).
    """
    h, w = gray.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    x_frac = cfg["alignment"].get("roi_center_x_frac", 0.75)
    y_frac = cfg["alignment"].get("roi_center_y_frac", 0.85)

    x0 = int(w * (1 - x_frac) / 2)
    x1 = int(w * (1 - (1 - x_frac) / 2))
    y0 = int(h * (1 - y_frac) / 2)
    y1 = int(h * (1 - (1 - y_frac) / 2))
    mask[y0:y1, x0:x1] = 255

    # Optional brightness mask
    if cfg["alignment"].get("use_brightness_mask", False):
        thresh_pct = cfg["alignment"].get("brightness_mask_percentile", 60)
        threshold = np.percentile(gray, thresh_pct)
        bright_mask = (gray >= threshold).astype(np.uint8) * 255
        # Erode to avoid selecting glare pixels at edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        bright_mask = cv2.erode(bright_mask, kernel, iterations=1)
        mask = cv2.bitwise_and(mask, bright_mask)

    return mask


def _keypoint_roi_fraction(kps: list, mask: np.ndarray) -> float:
    """
    Return fraction of keypoints that fall inside the ROI mask.
    Used for diagnostic logging.
    """
    if not kps:
        return 0.0
    inside = sum(1 for kp in kps if mask[int(kp.pt[1]), int(kp.pt[0])] > 0)
    return inside / len(kps)


# ---------------------------------------------------------------------------
# Primary alignment: ORB + BFMatcher + RANSAC
# ---------------------------------------------------------------------------

def _orb_align(img_a_gray: np.ndarray, img_b_gray: np.ndarray, cfg: dict,
               mask_a: np.ndarray = None, mask_b: np.ndarray = None):
    """
    Detect ORB keypoints on both images (optionally within ROI masks),
    match with BFMatcher+Lowe ratio test, compute RANSAC homography.

    Returns
    -------
    H           : np.ndarray (3x3) or None
    n_inliers   : int
    n_good      : int   matches after ratio test
    kp_a, kp_b  : keypoint lists (for diagnostics)
    """
    n_features = cfg["alignment"]["orb_n_features"]
    lowe_ratio = cfg["alignment"]["lowe_ratio"]
    ransac_thresh = cfg["alignment"]["ransac_threshold"]

    orb = cv2.ORB_create(nfeatures=n_features)
    kp_a, des_a = orb.detectAndCompute(img_a_gray, mask_a)
    kp_b, des_b = orb.detectAndCompute(img_b_gray, mask_b)

    if des_a is None or des_b is None or len(kp_a) < 4 or len(kp_b) < 4:
        logger.warning("ORB: insufficient keypoints (A=%d, B=%d)", len(kp_a), len(kp_b))
        return None, 0, 0, kp_a, kp_b

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = bf.knnMatch(des_b, des_a, k=2)

    good = []
    for pair in raw_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < lowe_ratio * n.distance:
                good.append(m)

    logger.debug("ORB matches after ratio test: %d", len(good))

    if len(good) < 4:
        return None, 0, len(good), kp_a, kp_b

    src_pts = np.float32([kp_b[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_a[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
    n_inliers = int(mask.sum()) if mask is not None else 0

    return H, n_inliers, len(good), kp_a, kp_b


def _ecc_align(img_a_gray: np.ndarray, img_b_gray: np.ndarray, cfg: dict) -> np.ndarray:
    """
    ECC (Enhanced Correlation Coefficient) alignment.

    Estimates an AFFINE warp by maximising correlation between the two images.
    More robust than phase correlation for slight rotation + scale + translation.
    Applied on center-cropped images to exclude background clutter.

    Falls back gracefully (returns None) if ECC does not converge.

    Returns 3x3 affine homography, or None on failure.
    """
    h, w = img_a_gray.shape[:2]
    # Use the center 60% x 70% crop for ECC
    x_frac = cfg["alignment"].get("roi_center_x_frac", 0.75)
    y_frac = cfg["alignment"].get("roi_center_y_frac", 0.85)
    # Tighten by ~20% for ECC to really focus on assembly
    x_frac = min(x_frac, 0.70)
    y_frac = min(y_frac, 0.75)

    x0 = int(w * (1 - x_frac) / 2)
    x1 = int(w * (1 - (1 - x_frac) / 2))
    y0 = int(h * (1 - y_frac) / 2)
    y1 = int(h * (1 - (1 - y_frac) / 2))

    crop_a = img_a_gray[y0:y1, x0:x1].astype(np.float32)
    crop_b = img_b_gray[y0:y1, x0:x1].astype(np.float32)

    # Downscale to speed up ECC (max 400px wide)
    scale = min(1.0, 400.0 / max(crop_a.shape))
    if scale < 1.0:
        crop_a = cv2.resize(crop_a, None, fx=scale, fy=scale)
        crop_b = cv2.resize(crop_b, None, fx=scale, fy=scale)

    warp_mode = cv2.MOTION_EUCLIDEAN  # rotation + translation only, more stable than affine
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-5)

    try:
        _, warp_matrix = cv2.findTransformECC(
            crop_a, crop_b, warp_matrix, warp_mode, criteria, None, 5
        )
    except cv2.error as e:
        logger.warning("ECC failed to converge: %s", e)
        return None

    # Scale warp back to full-image coordinates
    # The warp_matrix acts on the cropped + downscaled coords.
    # Convert: full_pt = crop_origin + crop_pt / scale
    # After ECC on crop: crop_pt_a = warp_matrix @ crop_pt_b
    # In full coords: full_pt_a - origin = warp_matrix[:,:2] @ (full_pt_b - origin) * scale / scale
    # Translation in full coords = translation_in_crop / scale + origin - warp_matrix[:,:2] @ origin
    tx_crop = warp_matrix[0, 2] / scale
    ty_crop = warp_matrix[1, 2] / scale
    R = warp_matrix[:, :2]
    tx_full = x0 + tx_crop - (R[0, 0] * x0 + R[0, 1] * y0) + x0 * 1  # corrected below
    ty_full = y0 + ty_crop - (R[1, 0] * x0 + R[1, 1] * y0) + y0 * 1

    # Correct formulation: warp in full-image coords
    # P_a = R * P_b + t   where P is in crop coords
    # P_a_full = P_a + origin = R*(P_b_full - origin) + t + origin
    #          = R*P_b_full + (t - R*origin + origin)
    origin = np.array([x0, y0], dtype=np.float32)
    t_crop = np.array([tx_crop, ty_crop], dtype=np.float32)
    t_full = t_crop - R @ origin + origin

    H_full = np.eye(3, dtype=np.float64)
    H_full[:2, :2] = R
    H_full[0, 2] = t_full[0]
    H_full[1, 2] = t_full[1]

    return H_full


def align_images(img_a: np.ndarray, img_b: np.ndarray, cfg: dict):
    """
    Warp img_b so it is geometrically aligned to img_a.

    Three-stage pipeline (each stage is a fallback to the next):
      1. ORB + RANSAC homography (best for rich-texture, well-matched shots)
      2. ECC Euclidean on center crop (robust to cluttered backgrounds)
      3. Phase correlation (pure translation, last resort)

    Parameters
    ----------
    img_a : np.ndarray  Reference image (BGR or grayscale).
    img_b : np.ndarray  Query image to warp (BGR or grayscale).
    cfg   : dict        Parsed config.json.

    Returns
    -------
    warped_b   : np.ndarray  img_b warped onto img_a's frame (same shape as img_a).
    confidence : str         "high" | "medium" | "low"
    method     : str         "homography" | "ecc" | "phase_correlation"
    H          : np.ndarray  3x3 homography used.
    """
    h_a, w_a = img_a.shape[:2]
    gray_a = _to_gray(img_a)
    gray_b = _to_gray(img_b)

    min_matches = cfg["alignment"]["min_good_matches"]
    confidence = "high"
    method = "homography"

    # --- Build ROI masks for ORB ---
    use_roi = cfg["alignment"].get("use_roi_mask", True)
    mask_a = _build_roi_mask(gray_a, cfg) if use_roi else None
    mask_b = _build_roi_mask(gray_b, cfg) if use_roi else None

    if use_roi and mask_a is not None:
        orb_tmp = cv2.ORB_create(nfeatures=500)
        kp_all_a, _ = orb_tmp.detectAndCompute(gray_a, None)
        kp_all_b, _ = orb_tmp.detectAndCompute(gray_b, None)
        frac_a = _keypoint_roi_fraction(kp_all_a, mask_a)
        frac_b = _keypoint_roi_fraction(kp_all_b, mask_b)
        logger.info(
            "ROI mask: A %.0f%% kpts in ROI, B %.0f%% kpts in ROI",
            frac_a * 100, frac_b * 100
        )

    # --- Stage 1: ORB homography ---
    H, n_inliers, n_good, kp_a, kp_b = _orb_align(gray_a, gray_b, cfg, mask_a, mask_b)
    orb_ok = (n_good >= min_matches) and not _is_degenerate_homography(H, img_a.shape)

    if orb_ok:
        logger.info("ORB homography: %d good matches, %d inliers.", n_good, n_inliers)
        warped_b = cv2.warpPerspective(img_b, H, (w_a, h_a))
        return warped_b, confidence, method, H

    logger.warning(
        "ORB alignment insufficient (good=%d, inliers=%d). Trying ECC.",
        n_good, n_inliers
    )

    # --- Stage 2: ECC on center crop ---
    H_ecc = _ecc_align(gray_a, gray_b, cfg)
    ecc_ok = H_ecc is not None and not _is_degenerate_homography(H_ecc, img_a.shape)

    if ecc_ok:
        logger.info("ECC alignment succeeded.")
        warped_b = cv2.warpPerspective(img_b, H_ecc, (w_a, h_a))
        return warped_b, "medium", "ecc", H_ecc

    logger.warning("ECC alignment failed. Falling back to phase correlation.")

    # --- Stage 3: Phase correlation (pure translation) ---
    H_phase = _phase_correlation_align(gray_a, gray_b)
    warped_b = cv2.warpPerspective(img_b, H_phase, (w_a, h_a))
    return warped_b, "low", "phase_correlation", H_phase


def test_alignment(img: np.ndarray, cfg: dict) -> float:
    """
    Synthetic alignment test.

    Applies a known rotation + translation + scale to img, then runs
    align_images() and measures the mean pixel reprojection error at
    the four image corners.

    Parameters
    ----------
    img : np.ndarray  Any BGR image.
    cfg : dict        Parsed config.json.

    Returns
    -------
    pixel_error : float  Mean corner reprojection error in pixels.
    """
    h, w = img.shape[:2]

    angle_deg = 3.0
    tx, ty = 15.0, 10.0
    scale = 0.97

    cx, cy = w / 2.0, h / 2.0
    rot = cv2.getRotationMatrix2D((cx, cy), angle_deg, scale)
    H_true = np.eye(3, dtype=np.float64)
    H_true[:2, :] = rot
    H_true[0, 2] += tx
    H_true[1, 2] += ty

    img_b_synth = cv2.warpPerspective(img, H_true, (w, h))

    _, confidence, method, H_est = align_images(img, img_b_synth, cfg)

    H_true_inv = np.linalg.inv(H_true)
    corners_b = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    gt_corners = cv2.perspectiveTransform(corners_b, H_true_inv)
    est_corners = cv2.perspectiveTransform(corners_b, H_est)

    errors = np.linalg.norm(gt_corners.reshape(4, 2) - est_corners.reshape(4, 2), axis=1)
    mean_error = float(errors.mean())

    logger.info(
        "Alignment test: method=%s, confidence=%s, mean_corner_error=%.3f px",
        method, confidence, mean_error
    )
    return mean_error
