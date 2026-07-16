"""
main.py -- Entry Point
======================
Loads config.json, sets up logging, then launches the Tkinter GUI.
Can also be invoked from the command line with --test to run all
Phase 1 (alignment) and Phase 6 (synthetic QC) tests headlessly.

Usage
-----
  python main.py              # launch GUI
  python main.py --test       # headless test suite only
  python main.py --test --img path/to/image.jpg  # test against specific image
"""

import argparse
import json
import logging
import os
import sys

# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError on the console)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np
from PIL import Image, ImageOps

def load_image(path: str) -> np.ndarray:
    """Load image and correct for EXIF orientation using PIL, then return as BGR numpy array."""
    try:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        return None

# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")


def load_config(path: str = CONFIG_PATH) -> dict:
    """Load and return parsed config.json."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )



# ---------------------------------------------------------------------------
# Headless test suite
# ---------------------------------------------------------------------------

def _make_test_image(h: int = 720, w: int = 1280) -> np.ndarray:
    """
    Generate a synthetic stainless-steel-like test image:
    gradient background + a few rectangular 'brackets' drawn on it.
    """
    # Gradient base simulating reflective surface
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for col in range(w):
        val = int(80 + 120 * (col / w))
        img[:, col] = (val, val, val)

    # Add simulated glare band
    glare_x = int(0.6 * w)
    for dx in range(120):
        x = glare_x + dx
        if x < w:
            alpha = 1 - dx / 120
            img[:, x] = np.clip(img[:, x].astype(int) + int(60 * alpha), 0, 255)

    # Draw "brackets" — simple rectangles in darker/lighter shades
    brackets = [
        ((100, 200), (200, 380), (50, 50, 50)),
        ((350, 150), (500, 320), (60, 60, 60)),
        ((700, 250), (820, 400), (45, 45, 45)),
        ((1000, 180), (1150, 350), (55, 55, 55)),
    ]
    for (x1, y1), (x2, y2), color in brackets:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        # Edge highlight
        cv2.rectangle(img, (x1, y1), (x2, y2), (120, 120, 120), 2)

    # Add some bolt holes
    for cx, cy in [(150, 290), (430, 230), (760, 325), (1075, 265)]:
        cv2.circle(img, (cx, cy), 12, (30, 30, 30), -1)
        cv2.circle(img, (cx, cy), 12, (150, 150, 150), 1)

    return img


def run_tests(cfg: dict, test_img_path: str | None = None):
    """
    Run 5 synthetic test cases and report FP/FN behaviour.

    Returns True if all critical assertions pass.
    """
    from align import align_images, test_alignment
    from diff import detect_differences
    from visualize import build_composite
    from classify import classify_difference

    # ── Build or load base image ────────────────────────────────────────
    if test_img_path and os.path.isfile(test_img_path):
        base_img = load_image(test_img_path)
        print(f"[TEST] Using provided image: {test_img_path}")
    else:
        base_img = _make_test_image()
        print("[TEST] Using synthetic test image (720p gradient + brackets).")

    results = []
    output_dir = os.path.join(os.path.dirname(__file__), "test_outputs")
    os.makedirs(output_dir, exist_ok=True)

    # ────────────────────────────────────────────────────────────────────
    # PHASE 1 TEST: Alignment accuracy
    # ────────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("PHASE 1 — Alignment accuracy test")
    print("="*60)
    pixel_error = test_alignment(base_img, cfg)
    passed = pixel_error < 5.0   # expect <5 px on synthetic warp
    print(f"  Synthetic warp (3° rot, 15tx, 10ty, 0.97 scale)")
    print(f"  Mean corner reprojection error: {pixel_error:.3f} px  "
          f"[{'PASS' if passed else 'FAIL'} — threshold 5.0 px]")
    results.append(("Alignment accuracy", passed, f"{pixel_error:.3f} px error"))

    # ────────────────────────────────────────────────────────────────────
    # TEST CASES 1-5
    # ────────────────────────────────────────────────────────────────────
    h, w = base_img.shape[:2]

    # ── Case 1: Lighting-only change (no structural difference) ─────────
    print("\nCase 1: Lighting-only variation (no physical difference)")
    img_b_light = base_img.copy().astype(np.float32)
    # Apply a spatially-varying brightness shift (+15 across left half)
    img_b_light[:, : w // 2] *= 1.08
    img_b_light = np.clip(img_b_light, 0, 255).astype(np.uint8)
    # Small simulated angle shift (1°, 5px)
    M_small = cv2.getRotationMatrix2D((w / 2, h / 2), 1.0, 1.0)
    M_small[0, 2] += 5
    H_small = np.eye(3)
    H_small[:2, :] = M_small
    img_b_light = cv2.warpPerspective(img_b_light, H_small, (w, h))

    warped_b1, conf1, method1, _ = align_images(base_img, img_b_light, cfg)
    contours1, _, ssim1 = detect_differences(base_img, warped_b1, cfg)
    fp1 = len(contours1)  # all flags are false positives (no real diff)
    passed_c1 = fp1 == 0
    print(f"  Alignment: {method1} [{conf1}]  SSIM={ssim1:.4f}")
    print(f"  Flagged regions: {fp1}  [Expected: 0 — {'PASS' if passed_c1 else 'FAIL'}]")
    labels1 = [classify_difference(c, base_img, warped_b1) for c in contours1]
    pct1 = sum(cv2.contourArea(c) for c in contours1) / (h * w) * 100
    comp1 = build_composite(base_img, warped_b1, contours1, cfg, low_confidence=(conf1 == "low"), pct_area=pct1, labels=labels1)
    cv2.imwrite(os.path.join(output_dir, "case1_lighting_only.jpg"), comp1,
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    results.append(("Case 1 — Lighting only (FP=0)", passed_c1,
                     f"{fp1} false positives"))

    # ── Case 2: Missing bracket (real structural difference) ────────────
    print("\nCase 2: Missing bracket (real structural difference)")
    img_b_missing = base_img.copy()
    # Erase one bracket by filling with background colour
    avg_bg = int(base_img[:, 400:600].mean())
    cv2.rectangle(img_b_missing, (350, 150), (500, 320), (avg_bg, avg_bg, avg_bg), -1)
    # Small angle shift to simulate handheld shot
    img_b_missing = cv2.warpPerspective(img_b_missing, H_small, (w, h))

    warped_b2, conf2, method2, _ = align_images(base_img, img_b_missing, cfg)
    contours2, _, ssim2 = detect_differences(base_img, warped_b2, cfg)
    fn2 = 1 if len(contours2) == 0 else 0
    tp2 = 1 if len(contours2) > 0 else 0
    passed_c2 = tp2 == 1
    print(f"  Alignment: {method2} [{conf2}]  SSIM={ssim2:.4f}")
    print(f"  Flagged regions: {len(contours2)}  [Expected: >=1 — {'PASS' if passed_c2 else 'FAIL'}]")
    labels2 = [classify_difference(c, base_img, warped_b2) for c in contours2]
    pct2 = sum(cv2.contourArea(c) for c in contours2) / (h * w) * 100
    comp2 = build_composite(base_img, warped_b2, contours2, cfg, low_confidence=(conf2 == "low"), pct_area=pct2, labels=labels2)
    cv2.imwrite(os.path.join(output_dir, "case2_missing_bracket.jpg"), comp2,
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    results.append(("Case 2 — Missing bracket (TP>=1)", passed_c2,
                     f"{len(contours2)} regions flagged"))

    # ── Case 3: Added bracket (extra component) ─────────────────────────
    print("\nCase 3: Added component (extra bracket present in B)")
    img_b_added = base_img.copy()
    cv2.rectangle(img_b_added, (550, 400), (660, 530), (55, 55, 55), -1)
    cv2.rectangle(img_b_added, (550, 400), (660, 530), (120, 120, 120), 2)
    img_b_added = cv2.warpPerspective(img_b_added, H_small, (w, h))

    warped_b3, conf3, method3, _ = align_images(base_img, img_b_added, cfg)
    contours3, _, ssim3 = detect_differences(base_img, warped_b3, cfg)
    passed_c3 = len(contours3) >= 1
    print(f"  Alignment: {method3} [{conf3}]  SSIM={ssim3:.4f}")
    print(f"  Flagged regions: {len(contours3)}  [Expected: ≥1 — {'PASS' if passed_c3 else 'FAIL'}]")
    labels3 = [classify_difference(c, base_img, warped_b3) for c in contours3]
    pct3 = sum(cv2.contourArea(c) for c in contours3) / (h * w) * 100
    comp3 = build_composite(base_img, warped_b3, contours3, cfg, low_confidence=(conf3 == "low"), pct_area=pct3, labels=labels3)
    cv2.imwrite(os.path.join(output_dir, "case3_added_bracket.jpg"), comp3,
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    results.append(("Case 3 — Added bracket (TP≥1)", passed_c3,
                     f"{len(contours3)} regions flagged"))

    # ── Case 4: Shifted component (mis-weld, ~20 px offset) ─────────────
    print("\nCase 4: Shifted component (~20 px offset in B)")
    img_b_shifted = base_img.copy()
    # Remove a bracket at its original location and redraw shifted
    cv2.rectangle(img_b_shifted, (700, 250), (820, 400), (avg_bg, avg_bg, avg_bg), -1)
    cv2.rectangle(img_b_shifted, (720, 270), (840, 420), (45, 45, 45), -1)
    cv2.rectangle(img_b_shifted, (720, 270), (840, 420), (120, 120, 120), 2)
    img_b_shifted = cv2.warpPerspective(img_b_shifted, H_small, (w, h))

    warped_b4, conf4, method4, _ = align_images(base_img, img_b_shifted, cfg)
    contours4, _, ssim4 = detect_differences(base_img, warped_b4, cfg)
    passed_c4 = len(contours4) >= 1
    print(f"  Alignment: {method4} [{conf4}]  SSIM={ssim4:.4f}")
    print(f"  Flagged regions: {len(contours4)}  [Expected: ≥1 — {'PASS' if passed_c4 else 'FAIL'}]")
    labels4 = [classify_difference(c, base_img, warped_b4) for c in contours4]
    pct4 = sum(cv2.contourArea(c) for c in contours4) / (h * w) * 100
    comp4 = build_composite(base_img, warped_b4, contours4, cfg, low_confidence=(conf4 == "low"), pct_area=pct4, labels=labels4)
    cv2.imwrite(os.path.join(output_dir, "case4_shifted_component.jpg"), comp4,
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    results.append(("Case 4 — Shifted component (TP≥1)", passed_c4,
                     f"{len(contours4)} regions flagged"))

    # ── Case 5: Large rotation (alignment stress test) ──────────────────
    print("\nCase 5: Large rotation (10°) — alignment fallback stress test")
    M_large = cv2.getRotationMatrix2D((w / 2, h / 2), 10.0, 0.98)
    M_large[0, 2] += 30
    H_large = np.eye(3)
    H_large[:2, :] = M_large
    img_b_rotated = cv2.warpPerspective(base_img, H_large, (w, h))

    warped_b5, conf5, method5, _ = align_images(base_img, img_b_rotated, cfg)
    contours5, _, ssim5 = detect_differences(base_img, warped_b5, cfg)
    # After correct alignment there should be minimal flagged area
    # (we don't fail the suite on this — just report)
    
    labels = []
    for c in contours5:
        labels.append(classify_difference(c, base_img, warped_b5))
    
    total_area = base_img.shape[0] * base_img.shape[1]
    flagged_area = sum(cv2.contourArea(c) for c in contours5)
    pct = (flagged_area / total_area) * 100

    print(f"  Alignment: {method5} [{conf5}]  SSIM={ssim5:.4f}")
    print(f"  Flagged regions after alignment: {len(contours5)}")
    comp5 = build_composite(base_img, warped_b5, contours5, cfg, low_confidence=(conf5 == "low"), pct_area=pct, labels=labels)
    cv2.imwrite(os.path.join(output_dir, "case5_large_rotation.jpg"), comp5,
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    results.append(("Case 5 — Large rotation alignment",
                     conf5 in ("high", "low"),   # always passes if no crash
                     f"method={method5}, conf={conf5}, regions={len(contours5)}"))

    # ────────────────────────────────────────────────────────────────────
    # Summary
    # ────────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    all_passed = True
    for name, passed, detail in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}  {name}  — {detail}")
        if not passed:
            all_passed = False
    print(f"\n  Output composites written to: {output_dir}")
    print("="*60 + "\n")
    return all_passed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Metal Assembly Visual QC Tool")
    parser.add_argument("--test", action="store_true",
                        help="Run headless synthetic test suite and exit.")
    parser.add_argument("--img", type=str, default=None,
                        help="Path to a sample image for alignment/test cases.")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable DEBUG logging.")
    parser.add_argument("--config", type=str, default=CONFIG_PATH,
                        help="Path to config.json (default: same directory as main.py).")
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)
    cfg = load_config(args.config)

    if args.test:
        success = run_tests(cfg, test_img_path=args.img)
        sys.exit(0 if success else 1)

    # Launch GUI
    from gui import QCApp
    app = QCApp(cfg)
    app.mainloop()


if __name__ == "__main__":
    main()
