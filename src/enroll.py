import face_recognition
import pickle
import os
import cv2
import sys
from locus.core_logic import calculate_novelties

# configuration
DATASET_PATH = "dataset"
ENCODING_FILE = "outputs/encodings.pickle"

def build_database():
    data = {"encodings": [], "names": [], "filenames": [], "geometry": []}
    if not os.path.exists(DATASET_PATH):
        print("path not found")
        sys.exit()
    stats = {"original": 0, "augmented": 0, "failed": 0}

    for root, dirs, files in os.walk(DATASET_PATH):
        for person_name in dirs:
            person_dir = os.path.join(root, person_name)
            for filename in sorted(os.listdir(person_dir)):
                if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                image_path = os.path.join(person_dir, filename)
                is_aug = "aug_" in filename

                try:
                    image = face_recognition.load_image_file(image_path)
                    boxes = face_recognition.face_locations(image, model="hog")
                    if not boxes:
                        stats["failed"] += 1
                        continue
                    encodings = face_recognition.face_encodings(image, boxes)
                    landmarks = face_recognition.face_landmarks(image, boxes)

                    if encodings:
                        data["encodings"].append(encodings[0])
                        data["names"].append(person_name)
                        data["filenames"].append(filename) 
                        
                        geo = calculate_novelties(landmarks[0])
                        data["geometry"].append(geo)

                        if is_aug: stats["augmented"] += 1
                        else: stats["original"] += 1

                except Exception as e:
                    print(f"Corrupt file {filename}: {e}")

    os.makedirs(os.path.dirname(ENCODING_FILE), exist_ok=True)
    with open(ENCODING_FILE, "wb") as f:
        f.write(pickle.dumps(data))
    
    print(f"model saved at {ENCODING_FILE}")

if __name__ == "__main__":
    build_database()