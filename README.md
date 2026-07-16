# Metal Assembly Visual QC Tool

This tool is a production-quality visual quality control (QC) application designed to compare two JPEG photos of a finished metal assembly and highlight physical differences (e.g., missing brackets, mis-welds, shifted components).

## Recommended Photo Capture Guidelines

For the most accurate and reliable results, this automated visual QC tool requires consistency between the Reference (Image A) and the Inspection (Image B) photos. Since the tool compares structural edges in 2D space, extreme 3D viewpoint changes (parallax) cannot be perfectly aligned and may result in false positive highlights. 

To prevent this, please adhere to standard industry practices for automated visual QC:
* **Fixed Camera Position:** Use a tripod or a designated, marked floor station. The camera distance, angle, and framing should be as identical as possible between the two shots.
* **Avoid Extreme Viewpoint Changes:** Do not move the camera horizontally or vertically between shots to expose different faces of 3D objects (e.g., shooting the left side of a post vs. the right side).
* **Consistent Lighting:** Keep shop lighting consistent. While the tool suppresses minor glare and reflection changes on stainless steel, extreme lighting shifts can introduce noise.

## Installation & Running

### Running the Standalone Executable
For end-users and technicians, a packaged standalone executable is provided in the `dist/` folder.
1. Double-click `dist/MetalQC.exe`.
2. The graphical interface will launch automatically (no Python installation required).
3. Click "Select Image 1 (Reference)" and "Select Image 2 (Inspection)" to load your JPEGs.
4. The tool will automatically align the images, detect differences, and display a side-by-side composite.
5. Click "Save Composite" to save the result.

### Running from Source
For in-house developers, you can run the tool directly from the Python source:
1. Ensure you have Python 3.9+ installed.
2. Create a clean virtual environment and install the required dependencies:
   ```bash
   python -m venv build_env
   build_env\Scripts\activate
   pip install opencv-python numpy Pillow
   ```
3. Run the application:
   ```bash
   python main.py
   ```

## Dependencies
* `opencv-python`: Core image processing, ORB/ECC alignment, homography, Canny edge detection, and SSIM computation.
* `numpy`: Matrix operations and image array manipulation.
* `Pillow`: Image loading (with robust EXIF orientation metadata correction) and UI compatibility.
* `tkinter`: Standard Python GUI library used for the application interface.

## How the tool reports low-confidence alignment
If the two images are too dissimilar (e.g., completely different parts, or a massive camera angle shift), the alignment stage may fail to find a reliable geometric match. 
When this happens:
1. An **orange warning banner** ("ALIGNMENT CONFIDENCE: LOW") will appear at the top of the output composite.
2. The red difference overlay will be **suppressed**.
This prevents the tool from highlighting the entire image in red due to a misaligned baseline. If a technician sees this orange banner, they should retake the inspection photo following the recommended capture guidelines to ensure it matches the reference angle.

## Configuration (config.json)
The `config.json` file exposes all tunable thresholds for the detection engine. In-house developers can adjust these without touching the source code:

* **Alignment (`alignment`)**
  * `orb_n_features`: Maximum keypoints to detect. Higher is more robust but slower.
  * `lowe_ratio`: Stricter matching ratio (e.g., 0.75) reduces false matches.
  * `use_roi_mask`: If true, restricts keypoints to the central region (defined by `roi_center_x_frac` and `roi_center_y_frac`), preventing background shop clutter from confusing the alignment.

* **Difference Detection (`difference`)**
  * `blur_kernel_size`: Applies a Gaussian blur (e.g., 15) before diffing to suppress slight parallax or edge misalignment noise.
  * `ssim_weight` & `edge_weight`: The final difference map is a weighted combination of structural similarity (SSIM) drops and Canny edge map differences.
  * `ssim_diff_cutoff`: Threshold for the combined difference map. Higher means more sensitive (flags smaller differences).
  * `canny_low` / `canny_high`: Edge detection thresholds.
  * `min_contour_area`: Minimum area (in pixels) for a difference to be flagged. This is the primary control for ignoring small dust/scratch noise (e.g., 4000).

* **Visualization (`visualization`)**
  * Controls the transparency (`overlay_alpha`), color (`box_color`), and size of the output composite.

## Project Structure

A compact overview of the repository to help contributors quickly understand where things live:

- **Top-level scripts:** `main.py`, `run_app.py`, `run_real_tests.py` — entry points for running the app or tests.
- **App modules:** `align.py`, `diff.py`, `classify.py`, `visualize.py`, `quantify.py` — core image-processing pipeline.
- **GUI / Web UI:** `gui.py` (desktop Tk UI) and the `templates/` + `static/` folders (Flask web UI assets).
- **Packaging:** `metal_qc.spec` — PyInstaller spec for building the standalone executable.
- **Configuration:** `config.json` — tunable detection thresholds and visualization options.

Notes and recommended housekeeping:
- Generated build artifacts and virtual environments should not be committed. Add them to `.gitignore` (see `.gitignore`).
- To update the standalone executable after making UI or template changes, rebuild using the `metal_qc.spec` file so the latest `templates/` and `static/` assets are bundled.
- If you'd like, I can remove or relocate the large `build/` and `build_env/` folders from the project to reduce clutter — confirm before I delete anything.
