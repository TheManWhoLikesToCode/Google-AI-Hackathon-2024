import time
import os
import cv2
import easyocr
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")


def read():
    start_time = time.time()  # Start time of the whole function

    section_times = []  # List to store times of each section

    reader = easyocr.Reader(["ch_sim", "en"])

    section_times.append(time.time())  # Mark end of initialization

    # Capture an image from the user's webcam
    cap = cv2.VideoCapture(0)

    camera_warmup_start = time.time()  # Start time for camera warm-up

    # Wait for the camera to warm up
    for _ in range(5):
        cap.read()

    section_times.append(time.time())  # Mark end of camera warm-up

    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Could not open webcam")

    capture_start = time.time()  # Start time for capturing image

    ret, frame = cap.read()
    cap.release()

    section_times.append(time.time())  # Mark end of image capture

    if not ret:
        raise RuntimeError("Failed to capture image from webcam")

    text_extraction_start = time.time()  # Start time for text extraction

    text = reader.readtext(frame)

    # Extract the text from the result with confidence threshold over 75
    text = [result[1] for result in text if result[2] > 0.50]

    section_times.append(time.time())  # Mark end of text extraction

    total_time = time.time() - start_time
    prev_time = start_time

    print("Time taken for each section and their percentage of total runtime:")
    sections = ["Initialization", "Camera Warm-Up", "Image Capture", "Text Extraction"]
    for i, (section, section_time) in enumerate(zip(sections, section_times)):
        section_duration = section_time - prev_time
        percentage = (section_duration / total_time) * 100
        print(f"{section}: {section_duration:.2f}s, {percentage:.2f}%")
        prev_time = section_time

    return text

print(read())
