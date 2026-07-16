import cv2
import numpy as np

def classify_difference(contour, img_a: np.ndarray, img_b: np.ndarray) -> str:
    """
    Classify a difference contour into:
    - Missing: Strong edges in A, weak edges in B.
    - Added: Weak edges in A, strong edges in B.
    - Shifted: Strong edges in both, but shifted in position.
    - Structural: Catch-all for texture/weld differences without clear addition/removal.
    """
    x, y, w, h = cv2.boundingRect(contour)
    
    # Extract patches
    patch_a = img_a[y:y+h, x:x+w]
    patch_b = img_b[y:y+h, x:x+w]
    
    # Convert to grayscale
    gray_a = cv2.cvtColor(patch_a, cv2.COLOR_BGR2GRAY) if patch_a.ndim == 3 else patch_a
    gray_b = cv2.cvtColor(patch_b, cv2.COLOR_BGR2GRAY) if patch_b.ndim == 3 else patch_b
    
    # Canny edges to measure structural density
    # Use fixed robust thresholds for classification to be independent of config thresholds
    edges_a = cv2.Canny(gray_a, 50, 150)
    edges_b = cv2.Canny(gray_b, 50, 150)
    
    # Create a mask for the exact contour within the bounding box
    contour_mask = np.zeros((h, w), dtype=np.uint8)
    # Offset contour to bounding box coordinates
    offset_contour = contour - [x, y]
    cv2.drawContours(contour_mask, [offset_contour], -1, 255, -1)
    
    area = cv2.countNonZero(contour_mask)
    if area == 0:
        return "Structural"
        
    # Calculate edge density (percentage of pixels that are edges)
    edges_a_masked = cv2.bitwise_and(edges_a, edges_a, mask=contour_mask)
    edges_b_masked = cv2.bitwise_and(edges_b, edges_b, mask=contour_mask)
    
    density_a = cv2.countNonZero(edges_a_masked) / area
    density_b = cv2.countNonZero(edges_b_masked) / area
    
    # Thresholds for classification (e.g. 2% edge density is a significant structure)
    density_thresh = 0.02
    
    if density_a > density_thresh and density_b < density_thresh:
        return "Missing"
    elif density_b > density_thresh and density_a < density_thresh:
        return "Added"
    elif density_a > density_thresh and density_b > density_thresh:
        # Both have structure. Are they shifted? 
        # Check if they are similar using a slightly expanded neighborhood in B
        exp = 20 # 20 px search radius
        y1, y2 = max(0, y-exp), min(img_a.shape[0], y+h+exp)
        x1, x2 = max(0, x-exp), min(img_a.shape[1], x+w+exp)
        
        search_patch_b = img_b[y1:y2, x1:x2]
        search_gray_b = cv2.cvtColor(search_patch_b, cv2.COLOR_BGR2GRAY) if search_patch_b.ndim == 3 else search_patch_b
        
        # Template match A's patch inside B's expanded search area
        # We need to make sure the patch isn't larger than the search area
        if search_gray_b.shape[0] >= gray_a.shape[0] and search_gray_b.shape[1] >= gray_a.shape[1]:
            res = cv2.matchTemplate(search_gray_b, gray_a, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            
            # If we found a strong match (>= 0.6) and it's not at the center (exp, exp), it's shifted
            if max_val >= 0.6:
                # Actual offset
                dy = max_loc[1] - (y - y1)
                dx = max_loc[0] - (x - x1)
                distance = np.sqrt(dx**2 + dy**2)
                
                if distance > 3.0: # More than 3 pixels shift
                    return "Shifted"
                    
        return "Structural"
        
    return "Structural"
