"""
run_real_tests.py -- Real-image pipeline test for both pairs.
Runs alignment + diff + visualization on both photo pairs and saves composites.
Prints alignment diagnostics per pair.
"""
import sys, json, os, cv2
sys.stdout.reconfigure(encoding='utf-8')

from align import align_images
from diff import detect_differences
from visualize import build_composite

with open('config.json') as f:
    cfg = json.load(f)

TEST_DIR = "Test Photos"
OUT_DIR  = "test_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

pairs = [
    ("Image 1.jpg", "Image 2.jpg", "pair1_elbow"),
    ("Photo1.jpg",  "Photo2.jpg",  "pair2_bracket"),
]

for fname_a, fname_b, label in pairs:
    print(f"\n{'='*60}")
    print(f"PAIR: {label}  ({fname_a} vs {fname_b})")
    print('='*60)

    img_a = cv2.imread(os.path.join(TEST_DIR, fname_a))
    img_b = cv2.imread(os.path.join(TEST_DIR, fname_b))
    print(f"  Image A size: {img_a.shape[1]}x{img_a.shape[0]}")
    print(f"  Image B size: {img_b.shape[1]}x{img_b.shape[0]}")

    warped_b, conf, method, H = align_images(img_a, img_b, cfg)
    print(f"  Alignment: {method} [{conf}]")
    print(f"  Homography:\n{H}")

    contours, mask, ssim_score = detect_differences(img_a, warped_b, cfg)
    print(f"  SSIM score: {ssim_score:.4f}")
    print(f"  Regions flagged: {len(contours)}")
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        print(f"    Region {i+1}: area={area:.0f}px^2, bbox=({x},{y} {w}x{h})")

    composite = build_composite(img_a, warped_b, contours, cfg, low_confidence=(conf == "low"))
    out_path = os.path.join(OUT_DIR, f"{label}_composite.jpg")
    cv2.imwrite(out_path, composite, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"  Saved: {out_path}")
