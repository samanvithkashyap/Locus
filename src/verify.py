import face_recognition
import pickle
import cv2
import numpy as np
import os
import sys
import csv
from collections import Counter
from datetime import datetime
from locus.core_logic import get_ear

#configuration
ENCODING_FILE = "outputs/encodings.pickle"
STUDENT_DB_FILE = "./data.csv"
TOLERANCE = 0.50       
BLINK_THRESH = 0.23    
CONSEC_FRAMES = 2      
SKIP_FRAMES = 3        

# lookup table
student_lookup = {}

def load_student_db():
    if not os.path.exists(STUDENT_DB_FILE):
        print(f"{STUDENT_DB_FILE} not found")
        return

    with open(STUDENT_DB_FILE, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 4: continue
            db_name = row[1].strip()
            student_lookup[db_name.lower()] = {
                'real_name': db_name,
                'usn': row[2].strip(), 
                'uni': row[3].strip()
            }

def get_student_details(detected_name):
    detected_lower = detected_name.lower()
    if detected_lower in student_lookup:
        return student_lookup[detected_lower]
    if '_' in detected_lower:
        parts = detected_lower.split('_')
        if parts[-1] in student_lookup:
            return student_lookup[parts[-1]]
        if parts[0] in student_lookup:
            return student_lookup[parts[0]]
    for key in student_lookup:
        if key in detected_lower:
            return student_lookup[key]

    return {'real_name': detected_name, 'usn': 'N/A', 'uni': 'N/A'}

def mark_attendance(name):
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    time_str = now.strftime('%H:%M:%S')
    filename = f"attendance_{date_str}.csv"
    details = get_student_details(name)
    real_name = details['real_name']
    usn = details['usn']
    uni = details['uni']

    if not os.path.exists(filename):
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Name", "USN", "University"])
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            if real_name in f.read(): return
    with open(filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([time_str, real_name, usn, uni])
        print(f"\n[SUCCESS] >>> MARKED: {real_name} | {usn} | {uni} <<<\n")

def run_live_system():
    load_student_db()
    if not os.path.exists(ENCODING_FILE):
        print(f"{ENCODING_FILE} not found.")
        sys.exit()

    with open(ENCODING_FILE, "rb") as f:
        db = pickle.loads(f.read())
    print("opening camera")
    video_capture = cv2.VideoCapture(0)
    
    blink_counter = {}     
    liveness_verified = {} 
    frame_count = 0
    
    current_boxes = []
    current_names = []
    current_landmarks = []

    while True:
        ret, frame = video_capture.read()
        if not ret: break
        if frame_count % (SKIP_FRAMES + 1) == 0:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            current_boxes = face_recognition.face_locations(rgb_small_frame, model="hog")
            encodings = face_recognition.face_encodings(rgb_small_frame, current_boxes)
            current_landmarks = face_recognition.face_landmarks(rgb_small_frame, current_boxes)
            
            current_names = []

            for encoding in encodings:
                distances = face_recognition.face_distance(db["encodings"], encoding)
                matches_indices = [i for i, dist in enumerate(distances) if dist < TOLERANCE]
                
                name = "Unknown"
                if len(matches_indices) > 0:
                    votes = Counter([db["names"][i] for i in matches_indices])
                    best_match, count = votes.most_common(1)[0]
                    if count >= 2: 
                        name = best_match
                
                current_names.append(name)

        frame_count += 1
        for name, landmarks in zip(current_names, current_landmarks):
            if name == "Unknown": continue

            if name not in blink_counter:
                blink_counter[name] = 0
                liveness_verified[name] = False

            left_ear = get_ear(landmarks['left_eye'])
            right_ear = get_ear(landmarks['right_eye'])
            avg_ear = (left_ear + right_ear) / 2.0

            if avg_ear < BLINK_THRESH:
                blink_counter[name] += 1
            else:
                if blink_counter[name] >= CONSEC_FRAMES:
                    liveness_verified[name] = True
                    mark_attendance(name)
                blink_counter[name] = 0 

        # draw HUD
        for (top, right, bottom, left), name in zip(current_boxes, current_names):
            top *= 4; right *= 4; bottom *= 4; left *= 4
            
            is_verified = liveness_verified.get(name, False)
            color = (0, 255, 0) if is_verified else (0, 0, 255) 
            
            # Fetch Clean Details for Display
            details = get_student_details(name)
            disp_usn = details['usn']
            disp_name = details['real_name']
            
            status = f"{disp_usn} [MARKED]" if is_verified else "BLINK NOW"
            if name == "Unknown":
                disp_name = "Unknown"
                color = (0, 0, 255)

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, disp_name, (left + 6, bottom - 20), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
            
            if name != "Unknown":
                cv2.putText(frame, status, (left + 6, bottom - 5), 
                            cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 1)

        cv2.imshow('Face Auth', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_live_system()