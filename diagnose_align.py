"""
diagnose_align.py -- Diagnostic: show keypoint distribution before/after ROI mask
for Pair 2 (cluttered background). Saves annotated images to test_outputs/.
"""
import sys, json, os, cv2, numpy as np
sys.stdout.reconfigure(encoding='utf-8')

with open('config.json') as f:
    cfg = json.load(f)

TEST_DIR = "Test Photos"
OUT_DIR  = "test_outputs"

# Load the two bracket photos
img_a = cv2.imread(os.path.join(TEST_DIR, "Photo1.jpg"))
img_b = cv2.imread(os.path.join(TEST_DIR, "Photo2.jpg"))
h, w = img_a.shape[:2]
gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create(nfeatures=1000)

# --- Test 1: No mask ---
kp_a_all, _ = orb.detectAndCompute(gray_a, None)
kp_b_all, _ = orb.detectAndCompute(gray_b, None)
print(f"No mask: A={len(kp_a_all)} kpts, B={len(kp_b_all)} kpts")

# --- Test 2: Center-crop 75%x85% ---
from align import _build_roi_mask
cfg70 = dict(cfg)
cfg70['alignment'] = dict(cfg['alignment'])
cfg70['alignment']['roi_center_x_frac'] = 0.75
cfg70['alignment']['roi_center_y_frac'] = 0.85
cfg70['alignment']['use_brightness_mask'] = False
mask_a70 = _build_roi_mask(gray_a, cfg70)
mask_b70 = _build_roi_mask(gray_b, cfg70)
kp_a_70, _ = orb.detectAndCompute(gray_a, mask_a70)
kp_b_70, _ = orb.detectAndCompute(gray_b, mask_b70)
print(f"ROI 75%x85%: A={len(kp_a_70)} kpts, B={len(kp_b_70)} kpts")

# --- Test 3: Tighter 55%x60% ---
cfg55 = dict(cfg)
cfg55['alignment'] = dict(cfg['alignment'])
cfg55['alignment']['roi_center_x_frac'] = 0.55
cfg55['alignment']['roi_center_y_frac'] = 0.60
cfg55['alignment']['use_brightness_mask'] = False
mask_a55 = _build_roi_mask(gray_a, cfg55)
mask_b55 = _build_roi_mask(gray_b, cfg55)
kp_a_55, _ = orb.detectAndCompute(gray_a, mask_a55)
kp_b_55, _ = orb.detectAndCompute(gray_b, mask_b55)
print(f"ROI 55%x60%: A={len(kp_a_55)} kpts, B={len(kp_b_55)} kpts")

# Save annotated image showing all three masks for Photo1
vis = img_a.copy()
# Draw all keypoints red
for kp in kp_a_all:
    cv2.circle(vis, (int(kp.pt[0]), int(kp.pt[1])), 3, (0, 0, 255), -1)
# Draw ROI 75% in yellow
x0 = int(w * (1-0.75)/2); x1 = int(w * (1-(1-0.75)/2))
y0 = int(h * (1-0.85)/2); y1 = int(h * (1-(1-0.85)/2))
cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 255), 3)
# Draw ROI 55% in green
x0b = int(w * (1-0.55)/2); x1b = int(w * (1-(1-0.55)/2))
y0b = int(h * (1-0.60)/2); y1b = int(h * (1-(1-0.60)/2))
cv2.rectangle(vis, (x0b, y0b), (x1b, y1b), (0, 255, 0), 3)
cv2.putText(vis, "Red=all kpts  Yellow=75%x85% ROI  Green=55%x60% ROI",
            (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
cv2.imwrite(os.path.join(OUT_DIR, "pair2_keypoint_diagnosis.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
print(f"Saved: {OUT_DIR}/pair2_keypoint_diagnosis.jpg")

# Also try running alignment with 55% ROI and report result
cfg55['alignment']['use_roi_mask'] = True
from align import align_images
warped_b55, conf55, method55, H55 = align_images(img_a, img_b, cfg55)
from diff import detect_differences
contours55, _, ssim55 = detect_differences(img_a, warped_b55, cfg)
print(f"\nWith 55%x60% ROI: alignment={method55} [{conf55}], SSIM={ssim55:.4f}, regions={len(contours55)}")
print(f"H=\n{H55}")
