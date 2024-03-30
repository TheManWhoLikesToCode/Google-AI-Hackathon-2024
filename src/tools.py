import time
import os
import cv2
import easyocr
from dotenv import load_dotenv

from imagehash import phash
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

load_dotenv()

reader = easyocr.Reader(["ch_sim", "en"])
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# Changes to me made using vidgear library we want to stablize the videofeed to help in text extraction to do this we must first change the


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

    camera_warmup_start = time.time()  # Start time for camera warm-up

    # Wait for the camera to warm up
    for _ in range(3):
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


def see():
    """
    Capture frames from the webcam in real-time and select distinct frames based on perceptual hash similarity.
    """
    start_time = time.time()
    output_directory = "selected_frames"
    os.makedirs(output_directory, exist_ok=True)

    # Create a capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Could not open webcam")

    n_frames = 60  # Number of frames to process
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    selected_frames = []
    previous_hashes = []
    hash_threshold = 5  # Adjust this threshold as needed

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

    # Releasing the video capture object to free the space captured
    cap.release()

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Real-time Frame Capture: {execution_time:.2f} seconds")

    print(f"Total key frames based on the threshold chosen: {len(selected_frames)}")

    # Selected frames display
    fig, axs = plt.subplots(1, len(selected_frames), figsize=(30, 10))
    for i, frame in enumerate(selected_frames):
        axs[i].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        axs[i].set_title(f"Selected Frame {i}")
        axs[i].axis("off")
    plt.show()

    return selected_frames


if __name__ == "__main__":
    see()
