"""
Author: Jaydin
Date: 03-27-2024
Description: REST API for my Google AI Hackathon Project Submission
To Run: `uvicorn main:app --reload`
"""
from fastapi import FastAPI, File, Header, Query, UploadFile, Form
from config import generation_config, safety_settings
import google.generativeai as genai
from io import BytesIO
import base64
from pprint import pprint

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/conversation")
async def conversation(prompt: str = Form(...), auth_token: str = Header(...), model: str = Query(...)):
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

    # Define function declarations for reading and magnifying modes
    tools = [
        {
            "function_declarations": [
                {
                    "name": "reading_mode",
                    "description": "Extract the text from the provided image",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "image": {
                                "type": "string",
                                "description": "The base64-encoded image data"
                            }
                        },
                        "required": ["image"]
                    }
                },
                {
                    "name": "magnifying_mode",
                    "description": "Describe the objects in the provided image in detail",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "image": {
                                "type": "string",
                                "description": "The base64-encoded image data"
                            }
                        },
                        "required": ["image"]
                    }
                }
            ]
        }
    ]

    # Generate the response using the model with function calling
    response = model.generate_content(prompt, tools=tools)
    print("Full response:")
    pprint(response)

    # Print specific parts of the response
    print("\nResponse parts:")
    for i, part in enumerate(response.parts, start=1):
        print(f"Part {i}:")
        pprint(part)
        print()  # Empty line for separation


    if hasattr(response.candidates[0].content.parts[0], "function_call"):
        function_call = response.candidates[0].content.parts[0].function_call
        print("Function call details:")
        pprint(function_call)

        function_name = function_call.name
        if function_name == "reading_mode" or function_name == "magnifying_mode":
            return {"message": "Please upload an image for " + function_name}
    else:
        return {"response": response.candidates[0].content.parts[0].text.strip()}


@app.post("/capture_image")
async def capture_image(file: UploadFile = File(...), mode: str = Form(...)):
    # Save the uploaded image file to memory
    image_data = await file.read()
    image_buffer = BytesIO(image_data)

    # Convert the image data to base64
    base64_image = base64.b64encode(image_buffer.getvalue()).decode("utf-8")

    # Prepare the function call based on the mode
    if mode == "reading_mode":
        function_call = {
            "name": "reading_mode",
            "args": {
                "image": base64_image
            }
        }
    elif mode == "magnifying_mode":
        function_call = {
            "name": "magnifying_mode",
            "args": {
                "image": base64_image
            }
        }
    else:
        return {"Error": "Invalid mode"}

    # Load Model
    model = genai.GenerativeModel(
        model_name=model,
        generation_config=generation_config,
        safety_settings=safety_settings
    )

    # Generate the response using the model with the captured image
    response = model.generate_content(function_call)

    if mode == "reading_mode":
        return {"text": response.text.strip()}
    elif mode == "magnifying_mode":
        return {"description": response.text.strip()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
