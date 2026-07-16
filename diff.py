"""
diff.py -- Difference Detection Module
======================================
Detects meaningful structural differences between two aligned images.

Pipeline
--------
1. CLAHE normalization on both images to suppress global lighting variations.
2. SSIM per-pixel difference map (primary signal for texture/intensity changes).
3. Canny edge-XOR map (secondary signal for structural/geometric changes).
4. Weighted combination -> binary mask after threshold.
5. Zero-border mask: exclude warp-padded black margins from result.
6. Morphological open (remove speckle) -> close (merge nearby regions).
7. Contour extraction -> discard contours below min_area.

Public API
----------
detect_differences(img_a, warped_b, cfg)
    -> (contours, diff_mask, ssim_score)

Note on SSIM implementation
---------------------------
SSIM is implemented directly with NumPy/OpenCV following Wang et al. (2004)
"Image Quality Assessment: From Error Visibility to Structural Similarity".
Constants K1=0.01, K2=0.03 match the skimage defaults exactly.
This removes the scikit-image dependency while preserving identical output.

Note on CLAHE pre-processing
-----------------------------
Contrast Limited Adaptive Histogram Equalization (CLAHE) is applied per-channel
before SSIM to suppress spatially-uniform luminance shifts (lighting variation,
exposure change) that would otherwise generate false positives on metal surfaces.
This does NOT suppress structural differences (shape, edges remain unchanged).
"""

import cv2
import numpy as np
import logging

from align import _build_roi_mask

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.copy()


def _clahe_normalize(gray: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE to suppress global/regional lighting variations.

    clip_limit=2.0 and tileGridSize=8x8 are conservative choices that
    suppress brightness gradients while preserving fine structural detail.

    Returns uint8 normalized grayscale image.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _valid_pixel_mask(warped_b_gray: np.ndarray, erode_px: int = 8) -> np.ndarray:
    """
    Build a binary mask that is 255 where warped_b has valid (non-zero-padded)
    pixels, and 0 in the black border introduced by cv2.warpPerspective.

    erode_px: extra erosion inward from the border to avoid edge-interpolation
              artefacts that SSIM would catch as false differences.

    Returns uint8 mask, same shape as warped_b_gray.
    """
    # Consider pixels >2 as valid (allows for very dark but real image content)
    valid = (warped_b_gray > 2).astype(np.uint8) * 255
    if erode_px > 0:
        kernel = np.ones((erode_px, erode_px), np.uint8)
        valid = cv2.erode(valid, kernel, iterations=1)
    return valid


def _ssim_diff_map(gray_a: np.ndarray, gray_b: np.ndarray, cfg: dict):
    """
    Compute per-pixel SSIM between two grayscale uint8 images.

    Implements Wang et al. (2004) with a Gaussian sliding window, matching
    the default behaviour of skimage.metrics.structural_similarity(full=True).

    Returns
    -------
    score    : float        Mean SSIM over the image (1 = identical).
    diff_map : np.ndarray   Per-pixel difference map, float32 in [0, 1].
                            0 = no difference, 1 = maximum difference.
    """
    win_size = cfg["difference"]["ssim_win_size"]
    data_range = 255.0
    K1, K2 = 0.01, 0.03        # Wang et al. defaults
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    # Build a Gaussian kernel (sigma=1.5 matches skimage default)
    sigma = 1.5
    kernel_1d = cv2.getGaussianKernel(win_size, sigma)
    kernel_2d = kernel_1d @ kernel_1d.T   # outer product -> 2-D Gaussian

    # Convert to float32 for convolution
    fa = gray_a.astype(np.float32)
    fb = gray_b.astype(np.float32)

    def _conv(img):
        """2-D Gaussian-weighted local mean via filter2D."""
        return cv2.filter2D(img, -1, kernel_2d, borderType=cv2.BORDER_REFLECT)

    mu_a   = _conv(fa)
    mu_b   = _conv(fb)
    mu_aa  = _conv(fa * fa)
    mu_bb  = _conv(fb * fb)
    mu_ab  = _conv(fa * fb)

    sigma_aa = mu_aa - mu_a * mu_a
    sigma_bb = mu_bb - mu_b * mu_b
    sigma_ab = mu_ab - mu_a * mu_b

    # Per-pixel SSIM map (Wang et al. eq. 13)
    numerator   = (2.0 * mu_a * mu_b + C1) * (2.0 * sigma_ab + C2)
    denominator = (mu_a**2 + mu_b**2 + C1) * (sigma_aa + sigma_bb + C2)
    S = numerator / (denominator + 1e-10)

    score = float(S.mean())

    # Convert similarity -> difference, clip to [0, 1]
    # Using (1-S)/2 maps [-1,1] -> [0,1] uniformly, but real S is in [0.7,1]
    # so we use (1-S) directly for more spread, then clip.
    diff_map = np.clip(1.0 - S, 0.0, 1.0).astype(np.float32)
    return score, diff_map


def _edge_diff_map(gray_a: np.ndarray, gray_b: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Compute XOR of Canny edge maps. Returns float32 map in [0, 1].
    Detects structural/geometric differences (missing parts, shifted brackets)
    that SSIM alone might smooth over in uniform-colour metal regions.
    """
    low = cfg["difference"]["canny_low"]
    high = cfg["difference"]["canny_high"]

    edges_a = cv2.Canny(gray_a, low, high)
    edges_b = cv2.Canny(gray_b, low, high)

    # XOR highlights edges present in one image but not the other
    xor_map = cv2.bitwise_xor(edges_a, edges_b)

    # Dilate slightly so nearby edge differences merge into regions
    kernel = np.ones((3, 3), np.uint8)
    xor_dilated = cv2.dilate(xor_map, kernel, iterations=1)

    return (xor_dilated / 255.0).astype(np.float32)


def _combine_maps(ssim_map: np.ndarray, edge_map: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Weighted sum of SSIM diff map and edge diff map -> binary mask.

    Both inputs are float32 in [0, 1]. Combined map is thresholded at
    the SSIM cutoff to produce a binary uint8 mask.
    """
    w_ssim = cfg["difference"]["ssim_weight"]
    w_edge = cfg["difference"]["edge_weight"]
    cutoff = cfg["difference"]["ssim_diff_cutoff"]

    combined = w_ssim * ssim_map + w_edge * edge_map
    # Normalise combined to [0, 1] by its theoretical maximum (w_ssim + w_edge)
    combined /= (w_ssim + w_edge)

    # Allow overriding the threshold when heavy blurring weakens the edge signal
    thresh_override = cfg["difference"].get("combined_threshold_override", None)
    if thresh_override is not None:
        threshold = thresh_override
    else:
        threshold = 1.0 - cutoff

    binary = (combined > threshold).astype(np.uint8) * 255
    return binary


def _morphological_clean(binary: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Apply morphological opening (remove speckle) then closing (fill gaps/merge regions).
    """
    k_size = cfg["difference"]["morph_kernel_size"]
    open_iter = cfg["difference"]["morph_open_iterations"]
    close_iter = cfg["difference"]["morph_close_iterations"]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=open_iter)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
    return cleaned


def _extract_contours(mask: np.ndarray, cfg: dict):
    """
    Extract external contours from binary mask and filter by minimum area.

    Returns list of contours (each is np.ndarray of shape [N,1,2]).
    """
    min_area = cfg["difference"]["min_contour_area"]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = [c for c in contours if cv2.contourArea(c) >= min_area]
    logger.debug(
        "Contours: total=%d, after area filter (>=%d px): %d",
        len(contours), min_area, len(filtered)
    )
    return filtered


def _assembly_foreground_mask(gray: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Build a foreground mask of the metal assembly to exclude background shop clutter.
    Uses Canny edge density + morphological closing to find the main central object,
    then returns its convex hull restricted by the alignment ROI.
    """
    # 1. Edge detection (strong structural features)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    
    # 2. Aggressive closing to merge into a solid foreground blob
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    closed = cv2.dilate(closed, kernel, iterations=2)
    
    # 3. Find largest contour
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        # Use convex hull to cover the whole assembly cleanly
        hull = cv2.convexHull(contours[0])
        cv2.drawContours(mask, [hull], -1, 255, -1)
        
        # Restrict the hull to a tighter center ROI (e.g. 60%x65%) 
        # to ensure extreme background edges are ignored
        h, w = gray.shape[:2]
        roi = np.zeros((h, w), dtype=np.uint8)
        x_frac = min(cfg["alignment"].get("roi_center_x_frac", 0.75), 0.60)
        y_frac = min(cfg["alignment"].get("roi_center_y_frac", 0.85), 0.65)
        x0 = int(w * (1 - x_frac) / 2)
        x1 = int(w * (1 - (1 - x_frac) / 2))
        y0 = int(h * (1 - y_frac) / 2)
        y1 = int(h * (1 - (1 - y_frac) / 2))
        roi[y0:y1, x0:x1] = 255
        
        mask = cv2.bitwise_and(mask, roi)
    else:
        # Fallback to standard ROI
        mask = _build_roi_mask(gray, cfg)
        
    return mask

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_differences(img_a: np.ndarray, warped_b: np.ndarray, cfg: dict):
    """
    Detect structural differences between reference image A and aligned image B.

    Parameters
    ----------
    img_a    : np.ndarray  Reference image (BGR or grayscale).
    warped_b : np.ndarray  Aligned query image (same shape as img_a, BGR or gray).
    cfg      : dict        Parsed config.json.

    Returns
    -------
    contours   : list[np.ndarray]  Contours of flagged difference regions.
    diff_mask  : np.ndarray        Binary (uint8) cleaned difference mask.
    ssim_score : float             Overall SSIM score between the two images.
    """
    gray_a = _to_gray(img_a)
    gray_b = _to_gray(warped_b)

    blur_k = cfg["difference"].get("blur_kernel_size", 0)
    if blur_k > 0:
        gray_a = cv2.GaussianBlur(gray_a, (blur_k, blur_k), 0)
        gray_b = cv2.GaussianBlur(gray_b, (blur_k, blur_k), 0)

    # Ensure identical sizes (warping should guarantee this, but be defensive)
    if gray_a.shape != gray_b.shape:
        gray_b = cv2.resize(gray_b, (gray_a.shape[1], gray_a.shape[0]))

    # --- Build valid-pixel mask to exclude warp black borders ---
    # Regions padded to zero by warpPerspective are not real image content.
    valid_mask = _valid_pixel_mask(gray_b)
    
    # --- Restrict diff to assembly ROI (to ignore background clutter) ---
    use_roi = cfg["alignment"].get("use_roi_mask", True)
    if use_roi:
        # Segment the assembly to prevent background false positives
        roi_a = _assembly_foreground_mask(gray_a, cfg)
        roi_b = _assembly_foreground_mask(gray_b, cfg)
        # The region must be valid foreground in EITHER image to catch missing/added parts
        fg_mask = cv2.bitwise_or(roi_a, roi_b)
        valid_mask = cv2.bitwise_and(valid_mask, fg_mask)

    # --- CLAHE normalization to suppress global lighting/exposure variation ---
    # This is the key step for stainless steel images where lighting may differ
    # between the two shots. CLAHE normalizes local contrast without blurring edges.
    norm_a = _clahe_normalize(gray_a)
    norm_b = _clahe_normalize(gray_b)

    # --- Primary: SSIM diff map (on CLAHE-normalized images) ---
    ssim_score, ssim_map = _ssim_diff_map(norm_a, norm_b, cfg)
    logger.info("SSIM score: %.4f", ssim_score)

    # --- Secondary: Edge XOR map (on CLAHE-normalized images) ---
    edge_map = _edge_diff_map(norm_a, norm_b, cfg)

    # --- Combine ---
    raw_binary = _combine_maps(ssim_map, edge_map, cfg)

    # --- Apply valid-pixel mask: zero out warp borders ---
    raw_binary = cv2.bitwise_and(raw_binary, raw_binary, mask=valid_mask)

    # --- Morphological cleanup ---
    cleaned = _morphological_clean(raw_binary, cfg)

    # --- Contour extraction ---
    contours = _extract_contours(cleaned, cfg)

    logger.info("Differences flagged: %d region(s).", len(contours))
    return contours, cleaned, ssim_score
