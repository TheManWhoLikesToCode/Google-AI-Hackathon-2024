"""
Author: Jaydin
Date: 03-27-2024
Description: REST API for my Google AI Hackathon Project Submission

To Run:
`uvicorn main:app --reload`

"""

from fastapi import FastAPI, File, Header, Query, UploadFile, Form
from config import generation_config, safety_settings
import google.generativeai as genai
from io import BytesIO

app = FastAPI()


@app.get("/")
def read_root():
    """
    A function that handles the root endpoint.

    Returns:
        dict: A dictionary with the message "Hello: World".
    """
    return {"Hello": "World"}


@app.get("/modes")
def get_modes():
    """
    Retrieve the available modes.

    Returns:
        dict: A dictionary containing the available modes.
    """
    return {"Modes": ["Reading Mode", "Magnifying Mode", "Conversationalist Mode"]}


@app.post("/mode/{mode}")
async def process_image(mode: str, file: UploadFile = File(...), auth_token: str = Header(...), model: str = Query(...)):
    """
    Process the uploaded image based on the selected mode.

    Parameters:
        - mode (str): The selected mode for processing the image.
        - file (UploadFile): The uploaded image file (optional for Conversationalist Mode).
        - auth_token (str): The authentication token for API access.
        - model (str): The name of the generative model to use.

    Returns:
        - If mode is "Reading Mode", returns a dictionary with the extracted text from the image.
        - If mode is "Magnifying Glass", returns a dictionary with the detailed description of objects in the image.
        - If mode is "Conversationalist Mode", returns a dictionary with the generated response from gemini-pro.
        - If mode is invalid, returns a dictionary with an error message.

    Raises:
        - If the auth_token is invalid, returns a dictionary with an error message.
    """
    # Check if the auth_token is valid
    try:
        genai.configure(api_key=auth_token)
    except:
        return {"Error": "Invalid API"}

    # Load Model
    model = genai.GenerativeModel(
        model_name=model,
        generation_config=generation_config,
        safety_settings=safety_settings
    )

    # Save the uploaded image file to memory
    if file is not None:
        image_data = await file.read()
        image_buffer = BytesIO(image_data)

        # Prepare the image for the model
        image_parts = [
            {
                "mime_type": "image/jpeg",
                "data": image_buffer.getvalue()
            }
        ]

    # Prepare the prompt based on the selected mode
    if mode == "Reading Mode":
        prompt_parts = [
            "Extract the text from the provided image",
            "Image: ",
            image_parts[0],
            "Extracted Text: ",
        ]
    elif mode == "Magnifying Glass":
        prompt_parts = [
            "Describe the objects in the provided image in detail",
            "Image: ",
            image_parts[0],
            "Detailed Description: ",
        ]
    elif mode == "Conversationalist Mode":
        prompt_parts = [
            "You are now in a conversation with gemini-pro, an AI assistant. Please provide your message or question.",
            "User: ",
        ]
    else:
        return {"Error": "Invalid mode selected"}

    # Generate the response using the model
    response = model.generate_content(prompt_parts)

    if mode == "Reading Mode":
        return {"text": response.text.strip()}
    elif mode == "Magnifying Glass":
        return {"description": response.text.strip()}
    elif mode == "Conversationalist Mode":
        return {"response": response.text.strip()}