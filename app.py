import sys
import os
import io
import json
import base64
import logging
from collections import Counter
import cv2
import numpy as np
from PIL import Image, ImageOps
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge

from align import align_images
from diff import detect_differences
from visualize import build_composite
from classify import classify_difference

if getattr(sys, 'frozen', False):
    base_dir = sys._MEIPASS
else:
    base_dir = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, 
            static_folder=os.path.join(base_dir, 'static'), 
            template_folder=os.path.join(base_dir, 'templates'))
CORS(app)

# Increase max upload size to 20 MB to handle high-res images
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024 

logger = logging.getLogger(__name__)

CONFIG_PATH = os.path.join(base_dir, "config.json")

def load_base_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def _lerp(val1, val2, t):
    return val1 + (val2 - val1) * t

def scale_config_by_sensitivity(base_cfg: dict, sensitivity: int) -> dict:
    """
    Scales config.json parameters based on 1-10 sensitivity slider.
    1 = Least sensitive, 5 = Default (baseline), 10 = Most sensitive.
    """
    # Clone config
    cfg = json.loads(json.dumps(base_cfg))
    s = float(np.clip(sensitivity, 1, 10))
    
    if s == 5.0:
        ssim = 0.92
        area = 4000
    elif s < 5.0:
        t = (s - 1.0) / 4.0
        ssim = _lerp(0.85, 0.92, t)
        area = _lerp(8000, 4000, t)
    else:
        t = (s - 5.0) / 5.0
        ssim = _lerp(0.92, 0.97, t)
        area = _lerp(4000, 1000, t)
        
    cfg["difference"]["ssim_diff_cutoff"] = ssim
    cfg["difference"]["min_contour_area"] = int(area)
    
    return cfg

def _load_cv2_image(file_storage):
    """Safely load an uploaded image, handling EXIF rotation."""
    try:
        img_bytes = file_storage.read()
        if not img_bytes:
            return None
        pil_img = Image.open(io.BytesIO(img_bytes))
        pil_img = ImageOps.exif_transpose(pil_img)
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        logger.error(f"Image load error: {e}")
        return None

def _encode_b64_image(cv2_img):
    """Encode OpenCV BGR image to base64 JPEG string."""
    _, buffer = cv2.imencode('.jpg', cv2_img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return base64.b64encode(buffer).decode('utf-8')

@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(e):
    return jsonify({"error": "File size exceeds the 20MB limit. Please upload a smaller image."}), 413

@app.errorhandler(Exception)
def handle_exception(e):
    logger.exception("Internal Server Error")
    return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/compare', methods=['POST'])
def compare():
    if 'image_a' not in request.files or 'image_b' not in request.files:
        return jsonify({"error": "Both image_a and image_b must be provided."}), 400
        
    file_a = request.files['image_a']
    file_b = request.files['image_b']
    
    if file_a.filename == '' or file_b.filename == '':
        return jsonify({"error": "No selected file for one or both images."}), 400

    try:
        sensitivity = int(request.form.get('sensitivity', 5))
    except ValueError:
        return jsonify({"error": "Invalid sensitivity value. Must be integer between 1 and 10."}), 400
        
    img_a = _load_cv2_image(file_a)
    img_b = _load_cv2_image(file_b)
    
    if img_a is None or img_b is None:
        return jsonify({"error": "Failed to decode uploaded images. Please ensure they are valid JPEGs/PNGs."}), 400
        
    base_cfg = load_base_config()
    cfg = scale_config_by_sensitivity(base_cfg, sensitivity)
    
    # 1. Align
    warped_b, confidence, method, _ = align_images(img_a, img_b, cfg)
    low_conf = (confidence == "low")
    
    # 2. Difference
    contours, diff_mask, ssim_score = detect_differences(img_a, warped_b, cfg)
    
    if low_conf:
        contours = []
        labels = []
        pct = 0.0
    else:
        # 3. Classify
        labels = []
        for c in contours:
            labels.append(classify_difference(c, img_a, warped_b))
            
        total_area = img_a.shape[0] * img_a.shape[1]
        flagged_area = sum(cv2.contourArea(c) for c in contours)
        pct = (flagged_area / total_area) * 100
    
    # 4. Composite
    composite = build_composite(img_a, warped_b, contours, cfg, low_confidence=low_conf, pct_area=pct, labels=labels)
    
    # Heatmap of difference mask
    heatmap = cv2.applyColorMap(diff_mask, cv2.COLORMAP_JET)
    # Thumbnail for isolated diff view
    h, w = heatmap.shape[:2]
    scale = min(400/w, 300/h)
    heatmap_thumb = cv2.resize(heatmap, (int(w*scale), int(h*scale)))
    
    counts = Counter(labels)
    breakdown = [{"label": k, "count": v} for k, v in counts.items()]
    
    response_data = {
        "composite_b64": _encode_b64_image(composite),
        "diff_thumb_b64": _encode_b64_image(heatmap_thumb),
        "flagged_area_pct": pct,
        "total_differences": len(contours),
        "classifications": breakdown,
        "confidence": confidence,
        "method": method,
        "ssim_score": ssim_score
    }
    
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5731, debug=True)
