"""
Author: Jaydin
Date: 04/02/2024
Description: Tools
"""

import time
import os
import shutil
from pathlib import Path
import cv2
from paddleocr import PaddleOCR
from dotenv import load_dotenv

from tqdm import tqdm
import google.generativeai as genai
from imagehash import phash
from PIL import Image
from config import generation_config, safety_settings

load_dotenv()

reader = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)

# Load Model
model = genai.GenerativeModel(
    model_name="gemini-pro-vision",
    generation_config=generation_config,
    safety_settings=safety_settings,
)


def print_section_times(sections, section_times, start_time):
    total_time = time.time() - start_time
    prev_time = start_time

    print("Time taken for each section and their percentage of total runtime:")
    for _, (section, section_time) in enumerate(zip(sections, section_times)):
        section_duration = section_time - prev_time
        percentage = (section_duration / total_time) * 100
        print(f"{section}: {section_duration:.2f}s, {percentage:.2f}%")
        prev_time = section_time


def read():
    """
    Reads text from the user's webcam.

    This function captures an image from the user's webcam, performs text extraction on the captured image,
    and returns the extracted text.

    Returns:
        list: A list of strings containing the extracted text.

    Raises:
        RuntimeError: If the webcam fails to open or if the image capture fails.
    """
    start_time = time.time()  # Start time of the whole function

    section_times = []  # List to store times of each section

    section_times.append(time.time())  # Mark end of initialization

    # Capture an image from the user's webcam
    cap = cv2.VideoCapture(0)

    # Wait for the camera to warm up
    for _ in range(5):
        cap.read()

    section_times.append(time.time())  # Mark end of setup

    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Could not open webcam")

    ret, frame = cap.read()
    cap.release()

    section_times.append(time.time())  # Mark end of data capture

    if not ret:
        raise RuntimeError("Failed to capture image from webcam")

    # Use EasyOCR for text extraction
    result = reader.ocr(frame, cls=True)
    if result and result[0]:
        # Extract the text from the OCR result
        ocr_text = " ".join([line[1][0] for line in result[0]])
    else:
        ocr_text = ""

    section_times.append(time.time())  # Mark end of data processing

    # Convert the frame to JPEG format and get the byte data
    _, img_data = cv2.imencode(".jpg", frame)
    img_bytes = img_data.tobytes()

    # Pass the OCR text and the captured image to Gemini for further processing
    prompt_parts = [
        "Instructions: The following text was extracted from an image using OCR. Please process the text and provide a cleaned-up version without any hallucinations or additions.",
        ocr_text,
        "Processed Text:",
        {"mime_type": "image/jpeg", "data": img_bytes},
    ]
    try:
        response = model.generate_content(prompt_parts)
        gemini_text = response.text.strip()
    except ValueError as e:
        print("Error occurred while generating content:")
        print(str(e))
        gemini_text = ""

    section_times.append(time.time())  # Mark end of model generation

    sections = [
        "Initialization",
        "Setup",
        "Data Capture",
        "Data Processing",
        "Model Generation",
    ]
    print_section_times(sections, section_times, start_time)

    return gemini_text


def see():
    """
    Capture frames from the webcam in real-time and select
    distinct frames based on perceptual hash similarity.
    """
    start_time = time.time()
    section_times = []  # List to store times of each section

    section_times.append(time.time())  # Mark end of initialization

    output_directory = "selected_frames"
    os.makedirs(output_directory, exist_ok=True)

    # Create a capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Could not open webcam")

    n_frames = 60  # Number of frames to process

    selected_frames = []
    previous_hashes = []
    hash_threshold = 15  # Adjust this threshold as needed

    section_times.append(time.time())  # Mark end of setup

    for frame_idx in tqdm(range(n_frames), desc="Processing Frames"):
        ret, img = cap.read()
        if not ret:
            break

        # Calculate the perceptual hash of the current frame
        current_hash = phash(Image.fromarray(img))

        # Compare the current hash with the previous selected frame hashes
        if not previous_hashes or all(
            current_hash - prev_hash >= hash_threshold for prev_hash in previous_hashes
        ):
            selected_frames.append(img)
            previous_hashes.append(current_hash)

            # Saving the selected frame to the output directory
            frame_filename = os.path.join(
                output_directory, f"frame_{frame_idx:04d}.png"
            )
            cv2.imwrite(frame_filename, img)

    section_times.append(time.time())  # Mark end of data capture

    # Releasing the video capture object to free the space captured
    cap.release()

    print(f"Total key frames based on the threshold chosen: {len(selected_frames)}")

    image_parts = []
    for i in os.listdir("selected_frames"):
        image_path = os.path.join("selected_frames", i)
        image_data = Path(image_path).read_bytes()
        image_part = {"mime_type": "image/png", "data": image_data}
        image_parts.append(image_part)

    prompt_parts = [
        "Instructions: Consider the following images and provide a summary of what is shown across all the images.",
        "Summary:",
    ]
    prompt_parts.extend(image_parts)
    prompt_parts.append("Description:")

    section_times.append(time.time())  # Mark end of data processing

    try:
        response = model.generate_content(prompt_parts)
    except ValueError as e:
        print("Error occurred while generating content:")
        print(str(e))
        response = None

    section_times.append(time.time())  # Mark end of model generation

    shutil.rmtree(output_directory)

    sections = [
        "Initialization",
        "Setup",
        "Data Capture",
        "Data Processing",
        "Model Generation",
    ]
    print_section_times(sections, section_times, start_time)

    return response.text


if __name__ == "__main__":
    read()
    see()
