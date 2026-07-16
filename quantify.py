import cv2
import numpy as np
import json
from align import align_images
from diff import detect_differences, _assembly_foreground_mask

def quantify_pair(img_a_path, img_b_path, name):
    with open('config.json', 'r') as f:
        cfg = json.load(f)
        
    img_a = cv2.imread(img_a_path)
    img_b = cv2.imread(img_b_path)
    
    # Save foreground masks
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    mask_a = _assembly_foreground_mask(gray_a, cfg)
    cv2.imwrite(f'test_outputs/{name}_mask_A.jpg', mask_a)
    
    # Run alignment
    warped_b, conf, method, H_est = align_images(img_a, img_b, cfg)
    
    if conf == "low":
        print(f"[{name}] Alignment Confidence: LOW. Overlay suppressed. Flagged area = 0.00%")
        return
        
    # Run diff
    contours, diff_mask, ssim_score = detect_differences(img_a, warped_b, cfg)
    
    total_area = img_a.shape[0] * img_a.shape[1]
    flagged_area = sum(cv2.contourArea(c) for c in contours)
    pct = (flagged_area / total_area) * 100
    
    print(f"[{name}] Flagged Area: {flagged_area:.0f} px^2 ({pct:.2f}% of total frame {total_area} px^2)")

if __name__ == '__main__':
    print("Quantifying Pair 1 (Elbow):")
    quantify_pair("Test Photos/Image 1.jpg", "Test Photos/Image 2.jpg", "pair1_elbow")
    print("\nQuantifying Pair 2 (Bracket):")
    quantify_pair("Test Photos/Photo1.jpg", "Test Photos/Photo2.jpg", "pair2_bracket")
    print("\nSelf-test (Image 1 vs Image 1):")
    quantify_pair("Test Photos/Image 1.jpg", "Test Photos/Image 1.jpg", "self_test")
