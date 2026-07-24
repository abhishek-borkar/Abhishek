import face_recognition
import cv2
import os

# Load HOD image (reference)
if not os.path.exists("hod.jpg"):
    print("Error: hod.jpg file not found!")
    exit()

hod_image = face_recognition.load_image_file("hod.jpg")
hod_encodings = face_recognition.face_encodings(hod_image)

if not hod_encodings:
    print("Error: No face detected in hod.jpg!")
    exit()

hod_encoding = hod_encodings[0]

# Folder containing test images
test_folder = "test_images"

if not os.path.exists(test_folder):
    print(f"Error: {test_folder} folder not found!")
    exit()

# Create output folder for results
output_folder = "detection_results"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through all images
for file in os.listdir(test_folder):
    if file.endswith(('.jpg', '.png', '.jpeg')):
        image_path = os.path.join(test_folder, file)
        
        try:
            # Load test image
            test_image = face_recognition.load_image_file(image_path)
            test_encodings = face_recognition.face_encodings(test_image)
            face_locations = face_recognition.face_locations(test_image)
        except Exception as e:
            print(f"{file} → ERROR (Could not load image: {str(e)})")
            continue
        
        # Read image with OpenCV for drawing
        cv_image = cv2.imread(image_path)
        hod_present = False
        hod_match_locations = []
        
        # Compare faces
        for idx, encoding in enumerate(test_encodings):
            match = face_recognition.compare_faces([hod_encoding], encoding)
            if match[0]:
                hod_present = True
                hod_match_locations.append(idx)
        
        # Draw rectangles and labels
        for idx, (top, right, bottom, left) in enumerate(face_locations):
            if idx in hod_match_locations:
                # Green for HOD Present
                color = (0, 255, 0)
                label = "HOD: YES"
            else:
                # Red for other faces
                color = (0, 0, 255)
                label = "Other"
            
            # Draw rectangle
            cv2.rectangle(cv_image, (left, top), (right, bottom), color, 2)
            
            # Put label
            cv2.putText(cv_image, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Add result text at top
        if hod_present:
            result_text = "HOD Present"
            result_color = (0, 255, 0)
            print(f"{file} → YES (HOD Present)")
        else:
            result_text = "HOD Absent"
            result_color = (0, 0, 255)
            print(f"{file} → NO (HOD Absent)")
        
        cv2.putText(cv_image, result_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, result_color, 2)
        
        # Save result image
        output_path = os.path.join(output_folder, f"detected_{file}")
        cv2.imwrite(output_path, cv_image)
        
        # Display image
        cv2.imshow(file, cv_image)
        cv2.waitKey(2000)  # Display for 2 seconds
        cv2.destroyAllWindows()

print(f"\nAll results saved to '{output_folder}' folder")
