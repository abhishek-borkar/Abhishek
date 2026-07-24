"""
Real-time image viewer for detection results
"""
import cv2
import os
from pathlib import Path

OUTPUT_FOLDER = "detection_results"

if not os.path.exists(OUTPUT_FOLDER):
    print(f"[ERROR] {OUTPUT_FOLDER} folder not found!")
    exit()

# Get all images sorted
images = sorted([f for f in os.listdir(OUTPUT_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

if not images:
    print(f"[WARN] No images found in {OUTPUT_FOLDER}")
    exit()

print(f"\n[INFO] Found {len(images)} detection result images")
print("[INFO] Press any key to move to next image, 'q' or ESC to quit\n")

for idx, filename in enumerate(images, 1):
    filepath = os.path.join(OUTPUT_FOLDER, filename)
    
    # Read and display image
    img = cv2.imread(filepath)
    if img is None:
        print(f"[SKIP] Could not load {filename}")
        continue
    
    # Display with title showing progress
    window_name = f"[{idx}/{len(images)}] {filename}"
    cv2.imshow(window_name, img)
    
    # Get image dimensions for info
    h, w = img.shape[:2]
    print(f"  [{idx:2d}/{len(images)}] {filename:<50} | {w}×{h}px")
    
    # Wait for key press (0 = infinite wait, or timeout in ms)
    key = cv2.waitKey(0)
    
    # ESC (27) or 'q' to quit
    if key == 27 or key == ord('q'):
        print("\n[INFO] Viewer closed by user")
        break
    
    cv2.destroyAllWindows()

cv2.destroyAllWindows()
print("\n[DONE] All images displayed\n")
