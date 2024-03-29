"""
Author: Jaydin F
Date: 03/29/2024
Description: Client

"""

import os
import speech_recognition as sr
from gtts import gTTS
from dotenv import load_dotenv
import google.generativeai as genai
from tools import read
from config import generation_config, safety_settings

load_dotenv()

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")


def speak(text):
    """
    Converts the given text to speech and plays it.

    Args:
        text (str): The text to be converted to speech.

    Returns:
        None
    """
    tts = gTTS(text=text, lang="en")
    tts.save("output.mp3")
    # Use 'afplay' for macOS, 'mpg321' for Linux, or 'start' for Windows
    os.system("afplay output.mp3")
    os.remove("output.mp3")


def listen():
    """
    Listens to audio input from the microphone and uses speech recognition to convert it into text.

    Returns:
        str: The recognized text from the audio input.

    Raises:
        sr.UnknownValueError: If the speech recognition engine could not understand the audio input.
        sr.RequestError: If there was an error making the request to the speech recognition service.
    """
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
        return text
    except sr.UnknownValueError as e:
        print(f"UnknownValueError: {str(e)}")
        return None
    except sr.RequestError as e:
        print(f"RequestError: {str(e)}")
        return None


def main():
    """
    Main function that serves as the entry point of the program.
    It loads the model, listens for user input, generates a response,
    and handles image uploads if prompted.

    Returns:
        None
    """
    genai.configure(api_key=GOOGLE_API_KEY)

    # Load Model
    model = genai.GenerativeModel(
        model_name="gemini-pro",
        generation_config=generation_config,
        safety_settings=safety_settings,
        tools=[read],
    )
    chat = model.start_chat(enable_automatic_function_calling=True)

    while True:
        user_input = listen()
        if user_input is None:
            speak("Sorry, I did not understand that. Please try again.")
            continue
        if user_input.lower() == "exit":
            speak("Goodbye!")
            break

        # Generate the response using the model with function calling
        response = chat.send_message(user_input)
        speak(response.text)

        for content in chat.history:
            part = content.parts[0]
            print(content.role, "->", type(part).to_dict(part))
            print("-" * 80)


if __name__ == "__main__":
    main()
