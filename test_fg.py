import cv2
import numpy as np

def _assembly_foreground_mask(gray: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    closed = cv2.dilate(closed, kernel, iterations=2)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        cv2.drawContours(mask, [contours[0]], -1, 255, -1)
    return mask

def test_mask(path, out_path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = _assembly_foreground_mask(gray)
    # Overlay red
    overlay = img.copy()
    overlay[mask == 0] = (0, 0, 0)
    cv2.imwrite(out_path, overlay)

test_mask("Test Photos/Photo1.jpg", "test_outputs/fg_Photo1.jpg")
test_mask("Test Photos/Image 1.jpg", "test_outputs/fg_Image1.jpg")
