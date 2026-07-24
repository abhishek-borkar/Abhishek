"""
HOD Face Detection — Raspberry Pi 4B (Bookworm / Bullseye)
===========================================================
Run ONCE to set up, then run the detector:

    bash setup_pi4b.sh          # installs everything into a venv
    source venv/bin/activate    # activate venv
    python3 hod_detection_pi4b.py

Folder layout expected next to this script:
    hod.jpg          ← reference face of the HOD
    test_images/     ← folder with .jpg/.jpeg/.png images to check
Output saved to:
    detection_results/
"""

import os
import sys

# ── Limit OpenBLAS threads — critical on Pi 4B to avoid OOM kills ────────────
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import cv2
import face_recognition

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG  — change these if your files live elsewhere
# ─────────────────────────────────────────────────────────────────────────────
HOD_IMAGE_PATH   = "hod.jpg"
TEST_FOLDER      = "test_images"
OUTPUT_FOLDER    = "detection_results"

# HOG = CPU-based, works well on Pi 4B (4× ARM Cortex-A72)
# Switch to "cnn" only if you add a swap file and have ≥4 GB RAM model
FACE_MODEL       = "hog"

# Resize large images before encoding — reduces RAM and speeds up processing
MAX_DIMENSION    = 640          # longest side in pixels
TOLERANCE        = 0.55         # lower = stricter match (default 0.6)

FONT             = cv2.FONT_HERSHEY_SIMPLEX
# ─────────────────────────────────────────────────────────────────────────────


def downscale(image_bgr):
    """Proportionally downscale so the longest side <= MAX_DIMENSION."""
    h, w = image_bgr.shape[:2]
    longest = max(h, w)
    if longest <= MAX_DIMENSION:
        return image_bgr
    scale  = MAX_DIMENSION / longest
    new_w  = max(1, int(w * scale))
    new_h  = max(1, int(h * scale))
    return cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def load_hod_encoding():
    """Load the HOD reference image and return its face encoding."""
    if not os.path.exists(HOD_IMAGE_PATH):
        sys.exit(f"[ERROR] Reference image not found: '{HOD_IMAGE_PATH}'\n"
                 "        Place hod.jpg next to this script and try again.")

    # face_recognition works in RGB; OpenCV loads BGR
    raw_bgr   = cv2.imread(HOD_IMAGE_PATH)
    if raw_bgr is None:
        sys.exit(f"[ERROR] OpenCV could not read '{HOD_IMAGE_PATH}'. "
                 "Is the file a valid JPEG/PNG?")

    small_bgr = downscale(raw_bgr)
    rgb       = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)

    encodings = face_recognition.face_encodings(rgb, model=FACE_MODEL)
    if not encodings:
        sys.exit("[ERROR] No face detected in hod.jpg!\n"
                 "        Make sure the face is clearly visible and well-lit.")

    print(f"[OK]  HOD reference loaded — 1 face encoded from '{HOD_IMAGE_PATH}'")
    return encodings[0]


def annotate_and_save(file_path, hod_encoding):
    """
    Process one test image:
      • detect all faces
      • compare each to HOD encoding
      • draw colour-coded boxes + labels
      • save annotated result to OUTPUT_FOLDER
    Returns True if HOD is present, False if absent, None on error.
    """
    filename = os.path.basename(file_path)

    raw_bgr = cv2.imread(file_path)
    if raw_bgr is None:
        print(f"  [SKIP] Could not read file: {filename}")
        return None

    # Downscale for faster processing
    small_bgr = downscale(raw_bgr)
    rgb       = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)

    # Detect face locations first, then encode — more reliable on Pi
    locations  = face_recognition.face_locations(rgb, model=FACE_MODEL)
    encodings  = face_recognition.face_encodings(rgb, locations)

    hod_present       = False
    hod_match_indices = set()

    for idx, enc in enumerate(encodings):
        matches = face_recognition.compare_faces(
            [hod_encoding], enc, tolerance=TOLERANCE
        )
        if matches[0]:
            hod_present = True
            hod_match_indices.add(idx)

    # ── Draw bounding boxes ──────────────────────────────────────────────────
    for idx, (top, right, bottom, left) in enumerate(locations):
        if idx in hod_match_indices:
            color, label = (0, 255, 0), "HOD: YES"
        else:
            color, label = (0, 0, 255), "Other"

        cv2.rectangle(small_bgr, (left, top), (right, bottom), color, 2)
        label_y = max(top - 10, 15)
        cv2.putText(small_bgr, label, (left, label_y),
                    FONT, 0.55, color, 2, cv2.LINE_AA)

    # ── Status banner at top-left ────────────────────────────────────────────
    if hod_present:
        banner, b_color = "HOD: PRESENT", (0, 200, 0)
    else:
        banner, b_color = "HOD: ABSENT",  (0, 0, 220)

    # Semi-transparent dark background strip for readability
    overlay = small_bgr.copy()
    cv2.rectangle(overlay, (0, 0), (small_bgr.shape[1], 50), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, small_bgr, 0.55, 0, small_bgr)
    cv2.putText(small_bgr, banner, (10, 35),
                FONT, 1.0, b_color, 2, cv2.LINE_AA)

    # ── Save result ──────────────────────────────────────────────────────────
    out_path = os.path.join(OUTPUT_FOLDER, f"detected_{filename}")
    cv2.imwrite(out_path, small_bgr)

    return hod_present


def main():
    print("\n=== HOD Face Detector — Raspberry Pi 4B ===\n")

    hod_encoding = load_hod_encoding()

    if not os.path.isdir(TEST_FOLDER):
        sys.exit(f"[ERROR] Test folder not found: '{TEST_FOLDER}'\n"
                 "        Create the folder and add images, then retry.")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    exts  = {".jpg", ".jpeg", ".png"}
    files = sorted(
        f for f in os.listdir(TEST_FOLDER)
        if os.path.splitext(f)[1].lower() in exts
    )

    if not files:
        sys.exit(f"[WARN] No .jpg/.jpeg/.png images found in '{TEST_FOLDER}'.")

    print(f"[INFO] Found {len(files)} image(s) to process.\n")

    present_count = 0
    absent_count  = 0
    error_count   = 0

    for fname in files:
        src = os.path.join(TEST_FOLDER, fname)
        print(f"  Processing: {fname:<35}", end="", flush=True)

        result = annotate_and_save(src, hod_encoding)

        if result is None:
            print("→ ERROR / SKIPPED")
            error_count += 1
        elif result:
            print("→ HOD PRESENT ✓")
            present_count += 1
        else:
            print("→ HOD ABSENT  ✗")
            absent_count += 1

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"""
─────────────────────────────────────
 Results Summary
─────────────────────────────────────
  HOD Present  : {present_count}
  HOD Absent   : {absent_count}
  Errors/Skipped: {error_count}
  Total         : {len(files)}
─────────────────────────────────────
 Annotated images saved → '{OUTPUT_FOLDER}/'
─────────────────────────────────────
""")


if __name__ == "__main__":
    main()
