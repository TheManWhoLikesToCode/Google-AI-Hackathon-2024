"""
Author: Jaydin Freeman
Date: 03/29/2024
Description: This file contains the functions for the client-side tools of the application.

"""

import os
import base64
import cv2
import easyocr
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")


def read():
    """
    Reads text from an image captured by the user's webcam.

    Returns:
        A list of extracted text from the image.
    """
    reader = easyocr.Reader(["ch_sim", "en"])

    # Capture an image from the user's webcam
    cap = cv2.VideoCapture(0)

    # Wait for the camera to warm up
    for _ in range(5):
        cap.read()

    if not cap.isOpened():
        cap.release()
        raise Exception("Could not open webcam")

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise Exception("Failed to capture image from webcam")

    text = reader.readtext(frame)

    # Extract the text from the result with confidence threshold over 75
    text = [result[1] for result in text if result[2] > 0.50]

    return text