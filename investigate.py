import cv2
import numpy as np
import json
from align import align_images
from diff import _ssim_diff_map, _edge_diff_map, _combine_maps, _morphological_clean, _extract_contours, _to_gray, _clahe_normalize, _assembly_foreground_mask, _valid_pixel_mask

def test_approach(img_a, img_b, cfg, approach_name, blur=None, edge_weight=None, ssim_cutoff=None, morph_iters=None, morph_kernel=None):
    # Deep copy config
    test_cfg = json.loads(json.dumps(cfg))
    
    if edge_weight is not None:
        test_cfg["difference"]["edge_weight"] = edge_weight
        test_cfg["difference"]["ssim_weight"] = 1.0 - edge_weight
    if ssim_cutoff is not None:
        test_cfg["difference"]["ssim_diff_cutoff"] = ssim_cutoff
    if morph_iters is not None:
        test_cfg["difference"]["morph_close_iterations"] = morph_iters
    if morph_kernel is not None:
        test_cfg["difference"]["morph_kernel_size"] = morph_kernel

    gray_a = _to_gray(img_a)
    gray_b = _to_gray(img_b)
    
    if blur:
        gray_a = cv2.GaussianBlur(gray_a, blur, 0)
        gray_b = cv2.GaussianBlur(gray_b, blur, 0)
        
    valid_mask = _valid_pixel_mask(gray_b)
    roi_a = _assembly_foreground_mask(gray_a, test_cfg)
    roi_b = _assembly_foreground_mask(gray_b, test_cfg)
    fg_mask = cv2.bitwise_or(roi_a, roi_b)
    valid_mask = cv2.bitwise_and(valid_mask, fg_mask)

    norm_a = _clahe_normalize(gray_a)
    norm_b = _clahe_normalize(gray_b)

    ssim_score, ssim_map = _ssim_diff_map(norm_a, norm_b, test_cfg)
    edge_map = _edge_diff_map(norm_a, norm_b, test_cfg)

    raw_binary = _combine_maps(ssim_map, edge_map, test_cfg)
    raw_binary = cv2.bitwise_and(raw_binary, raw_binary, mask=valid_mask)
    cleaned = _morphological_clean(raw_binary, test_cfg)
    contours = _extract_contours(cleaned, test_cfg)

    total_area = gray_a.shape[0] * gray_a.shape[1]
    flagged_area = sum(cv2.contourArea(c) for c in contours)
    pct = (flagged_area / total_area) * 100

    from visualize import build_composite
    composite = build_composite(img_a, img_b, contours, test_cfg, low_confidence=False, pct_area=pct)
    safe_name = approach_name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus")
    cv2.imwrite(f"test_outputs/investigate_{safe_name}.jpg", composite)
    
    print(f"{approach_name:40s} | Contours: {len(contours):2d} | Flagged: {pct:5.2f}%")
    return pct, cleaned

if __name__ == "__main__":
    with open('config.json', 'r') as f:
        cfg = json.load(f)
        
    img_a = cv2.imread("Test Photos/Photo1.jpg")
    img_b = cv2.imread("Test Photos/Photo2.jpg")
    warped_b, conf, method, H_est = align_images(img_a, img_b, cfg)
    
    print("Testing Pair 2 (Bracket) Approaches:")
    print("-" * 70)
    
    # Baseline
    test_approach(img_a, warped_b, cfg, "Baseline (Current)", blur=None)
    
    # Blur testing
    test_approach(img_a, warped_b, cfg, "Blur (7x7)", blur=(7, 7))
    test_approach(img_a, warped_b, cfg, "Blur (15x15)", blur=(15, 15))
    test_approach(img_a, warped_b, cfg, "Blur (21x21)", blur=(21, 21))
    
    # Edge weight testing
    test_approach(img_a, warped_b, cfg, "High Edge Weight (0.8)", edge_weight=0.8)
    test_approach(img_a, warped_b, cfg, "Only Edge Map (1.0)", edge_weight=1.0)
    
    # Combo testing
    test_approach(img_a, warped_b, cfg, "Blur(11x11) + Edge(0.7)", blur=(11, 11), edge_weight=0.7)
    test_approach(img_a, warped_b, cfg, "Blur(15x15) + Edge(0.8)", blur=(15, 15), edge_weight=0.8)
    
    # SSIM threshold
    test_approach(img_a, warped_b, cfg, "Stricter SSIM Cutoff (0.75)", ssim_cutoff=0.75)
    test_approach(img_a, warped_b, cfg, "Blur(15x15) + Strict SSIM(0.75)", blur=(15, 15), ssim_cutoff=0.75)
    test_approach(img_a, warped_b, cfg, "Morph Iters=1", morph_iters=1)
    test_approach(img_a, warped_b, cfg, "Morph Iters=1, Kernel=5", morph_iters=1, morph_kernel=5)
    

def block_ssim(gray_a, gray_b, block_size=16):
    h, w = gray_a.shape
    diff_mask = np.zeros((h, w), dtype=np.uint8)
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            # Exclude partial edge blocks for simplicity, or handle them
            bx1, by1 = x + block_size, y + block_size
            if bx1 > w or by1 > h:
                continue
            
            block_a = gray_a[y:by1, x:bx1]
            block_b = gray_b[y:by1, x:bx1]
            
            score = cv2.matchTemplate(block_b, block_a, cv2.TM_CCOEFF_NORMED)[0][0]
            # normalized cross correlation is highly resilient to smooth lighting/parallax changes
            # if score is low, mark block as diff
            if score < 0.35: # Tune threshold
                diff_mask[y:by1, x:bx1] = 255
    return diff_mask

def test_blockwise(img_a, img_b, cfg):
    test_cfg = json.loads(json.dumps(cfg))
    gray_a = _to_gray(img_a)
    gray_b = _to_gray(img_b)
    
    valid_mask = _valid_pixel_mask(gray_b)
    roi_a = _assembly_foreground_mask(gray_a, test_cfg)
    roi_b = _assembly_foreground_mask(gray_b, test_cfg)
    fg_mask = cv2.bitwise_or(roi_a, roi_b)
    valid_mask = cv2.bitwise_and(valid_mask, fg_mask)
    
    diff_mask = block_ssim(gray_a, gray_b, block_size=16)
    diff_mask = cv2.bitwise_and(diff_mask, diff_mask, mask=valid_mask)
    
    cleaned = _morphological_clean(diff_mask, test_cfg)
    contours = _extract_contours(cleaned, test_cfg)

    total_area = gray_a.shape[0] * gray_a.shape[1]
    flagged_area = sum(cv2.contourArea(c) for c in contours)
    pct = (flagged_area / total_area) * 100

    from visualize import build_composite
    composite = build_composite(img_a, img_b, contours, test_cfg, low_confidence=False, pct_area=pct)
    cv2.imwrite(f"test_outputs/investigate_Blockwise_TM_CCOEFF.jpg", composite)
    
    print(f"{'Blockwise TM_CCOEFF':40s} | Contours: {len(contours):2d} | Flagged: {pct:5.2f}%")

    # Test blockwise NCC
    test_blockwise(img_a, warped_b, cfg)
