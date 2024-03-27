import os
import requests
import base64
import json
import speech_recognition as sr
from gtts import gTTS
from dotenv import load_dotenv


load_dotenv()

GOOGLE_API_KEY = os.environ.get("AUTH_TOKEN")
API_URL = os.environ.get("API_URL")


def send_request(endpoint, data=None, files=None):
    headers = {"Authorization": f"Bearer {GOOGLE_API_KEY}"}
    if files:
        response = requests.post(
            f"{API_URL}/{endpoint}", data=data, files=files, headers=headers
        )
    else:
        response = requests.post(
            f"{API_URL}/{endpoint}", data=data, headers=headers
        )
    return response.json()


def speak(text):
    tts = gTTS(text=text, lang='en')
    tts.save("output.mp3")
    # Use 'afplay' for macOS, 'mpg321' for Linux, or 'start' for Windows
    os.system("afplay output.mp3")
    os.remove("output.mp3")


def listen():
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300  # Adjust the value as needed
    recognizer.pause_threshold = 1.0  # Adjust the value as needed
    recognizer.dynamic_energy_threshold = True

    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)

    try:
        print("Recognizing...")
        text = recognizer.recognize_google(audio)
        print(f"User: {text}")
        return text
    except sr.UnknownValueError as e:
        print(f"UnknownValueError: {str(e)}")
        return ""
    except sr.RequestError as e:
        print(f"RequestError: {str(e)}")
        return ""


def main():
    while True:
        user_input = listen()
        if user_input.lower() == "exit":
            speak("Goodbye!")
            break

        data = {"prompt": user_input}
        response = send_request(f"conversation?model=gemini-pro", data=data)

        if "message" in response:
            speak(response["message"])
            image_path = input("Enter the path to the image: ")
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(
                    image_file.read()).decode("utf-8")
            mode = "reading_mode" if "reading" in response["message"].lower(
            ) else "magnifying_mode"
            files = {"file": (image_path, base64_image)}
            data = {"mode": mode}
            response = send_request("capture_image", data=data, files=files)
            if "text" in response:
                speak("Extracted Text:")
                speak(response["text"])
            elif "description" in response:
                speak("Detailed Description:")
                speak(response["description"])
        elif "response" in response:
            speak("Assistant: " + response["response"])
        else:
            speak("Sorry, an error occurred.")


if __name__ == "__main__":
    main()
