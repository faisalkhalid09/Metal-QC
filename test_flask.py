import requests
import time
import subprocess
import os

print("Starting Flask server for testing...")
proc = subprocess.Popen(["python", "run_app.py"])
time.sleep(3) # Wait for server to start

def test_pair(name, img_a_path, img_b_path, sensitivity):
    url = "http://localhost:5731/api/compare"
    with open(img_a_path, 'rb') as fa, open(img_b_path, 'rb') as fb:
        files = {
            'image_a': ('Photo1.jpg', fa, 'image/jpeg'),
            'image_b': ('Photo2.jpg', fb, 'image/jpeg')
        }
        data = {'sensitivity': sensitivity}
        
        try:
            resp = requests.post(url, files=files, data=data)
            if resp.status_code == 200:
                res = resp.json()
                print(f"[{name}] Sens={sensitivity}: Area={res['flagged_area_pct']:.2f}% | Diffs={res['total_differences']} | Conf={res['confidence']}")
            else:
                print(f"[{name}] Sens={sensitivity} FAILED: {resp.text}")
        except Exception as e:
            print(f"[{name}] Sens={sensitivity} ERROR: {e}")

try:
    print("\n--- Testing Pair 1 (Elbow) ---")
    test_pair("Pair 1", "Test Photos/Image 1.jpg", "Test Photos/Image 2.jpg", 1)
    test_pair("Pair 1", "Test Photos/Image 1.jpg", "Test Photos/Image 2.jpg", 5)
    test_pair("Pair 1", "Test Photos/Image 1.jpg", "Test Photos/Image 2.jpg", 10)
    
    print("\n--- Testing Pair 2 (Bracket) ---")
    test_pair("Pair 2", "Test Photos/Photo1.jpg", "Test Photos/Photo2.jpg", 1)
    test_pair("Pair 2", "Test Photos/Photo1.jpg", "Test Photos/Photo2.jpg", 5)
    test_pair("Pair 2", "Test Photos/Photo1.jpg", "Test Photos/Photo2.jpg", 10)
finally:
    proc.terminate()
    proc.wait()
    print("Server stopped.")
