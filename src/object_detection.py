""" At the command line, only need to run once to install the package via pip: $ pip install google-generativeai """
import google.generativeai as genai
from PIL import Image, ImageTk
from dotenv import load_dotenv
import tkinter as tk
import cv2
import io
import os

# Set the API key from the environment variable
load_dotenv()
api_key = os.environ.get("GOOGLE_API_KEY")

genai.configure(api_key=api_key)

models = []

for model in genai.list_models():
  if 'generateContent' in model.supported_generation_methods:
    models.append(model)

# Ask the user to select a model
print("Select a model:")
for i, model in enumerate(models):
    print(f"{i + 1}. {model.name}")

model_index = int(input("Enter the model number: ")) - 1


# Set up the model
generation_config = {
    "temperature": 0.9,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 1024,
}
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]
model = genai.GenerativeModel(model_name="gemini-pro-vision", generation_config=generation_config, safety_settings=safety_settings)

# Create the main window
window = tk.Tk()
window.title("Camera Stream")

# Create a label to display the video frames
video_label = tk.Label(window)
video_label.pack()

# Open the default camera (index 0)
cap = cv2.VideoCapture(0)

def update_frame():
    # Read a frame from the camera
    ret, frame = cap.read()
    if ret:
        # Convert the frame to RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create a PIL Image from the frame
        img = Image.fromarray(frame_rgb)
        
        # Convert the PIL Image to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes = img_bytes.getvalue()
        
        # Create the image part for the API
        image_part = {"mime_type": "image/jpeg", "data": img_bytes}
        
        # Create the prompt parts
        prompt_parts = [
            "What objects are in this image? Describe how they might be used.",
            "Objects: ",
            image_part,
            "Description: "
        ]
        
        # Generate the response from the API
        response = model.generate_content(prompt_parts)
        
        # Print the generated description
        print(response.text)
        
        # Create a PhotoImage object from the PIL Image
        photo = ImageTk.PhotoImage(image=img)
        
        # Update the video label with the new frame
        video_label.config(image=photo)
        video_label.image = photo
    
    # Schedule the next frame update
    window.after(30, update_frame)

# Start updating the frames
update_frame()

# Run the main window loop
window.mainloop()

# Release the camera
cap.release()