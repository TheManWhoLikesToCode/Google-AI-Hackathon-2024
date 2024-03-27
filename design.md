# FastAPI Application: Visual Assistance for the Visually Impaired

## Design Doc
1. Problem Statement
   - Visually impaired individuals face challenges in interacting with the world around them, particularly in reading text and identifying objects.

2. Proposed Solution
   - Develop a FastAPI application that leverages the Google Vision API to provide visual assistance to visually impaired users.
   - The application will offer two main modes: Reading Mode and Magnifying Glass.
   - Users will interact with the application using voice commands.

3. User Flow
   - User activates the application using a voice command.
   - Application prompts the user to select a mode (Reading Mode or Magnifying Glass).
   - User speaks the desired mode.
   - Application captures an image from the connected camera.
   - Image is sent to the FastAPI server for processing.
   - FastAPI server sends the image to the Google Vision API for analysis.
   - API response is processed and the relevant information is returned to the user through speech output.

4. Technology Stack
   - FastAPI: Web framework for building the API endpoints and server-side logic.
   - Google Vision API: Cloud-based service for image analysis and object detection.
   - SpeechRecognition: Library for voice command recognition.
   - Camera Integration: Capture images from a connected camera (webcam or Raspberry Pi camera module).
   - Text-to-Speech: Convert the processed information into speech output for the user.

5. Security Considerations
   - Store Google Vision API credentials using a `.env` file.

6. Error Handling and Resilience
   - Implement robust error handling for voice commands, image capture, and API requests.
   - Provide clear error messages and guidance to users through speech output.
   - Ensure the application gracefully handles network disruptions and API failures.

7. Accessibility
   - Follow accessibility guidelines and best practices for voice interactions and audio output.
   - Conduct user testing with visually impaired individuals to gather feedback and improve usability.

## Main Components
1. FastAPI Server
   - Handles API endpoints and request processing
   - Integrates with Google Vision API for image analysis
   - Provides endpoints for different modes (Reading Mode, Magnifying Glass)

2. Voice Interaction
   - Utilizes a speech recognition library (e.g., SpeechRecognition) for voice commands
   - Allows users to select modes and interact with the application using voice

3. Camera Integration
   - Captures images from a connected camera (e.g., webcam, Raspberry Pi camera module)
   - Sends captured images to the FastAPI server for processing

4. Google Vision API Integration
   - Authenticates with the Google Vision API using credentials stored in the `.env` file
   - Sends images to the Vision API for analysis (text detection, object detection)
   - Retrieves and processes the API response

## API Endpoints
1. `/modes` (GET)
   - Returns a list of available modes (Reading Mode, Magnifying Glass)

2. `/reading_mode` (POST)
   - Accepts an image file as input
   - Performs text detection using the Google Vision API
   - Returns the extracted text from the selected object

3. `/magnifying_glass` (POST)
   - Accepts an image file as input
   - Performs object detection using the Google Vision API
   - Returns a zoomed-in description of the detected objects

## Voice Interaction Flow
1. User activates the application using a voice command (e.g., "Start visual assistance")
2. Application prompts the user to select a mode (e.g., "Please say 'Reading Mode' or 'Magnifying Glass'")
3. User speaks the desired mode
4. Application captures an image from the connected camera
5. Image is sent to the FastAPI server for processing
6. FastAPI server sends the image to the Google Vision API for analysis
7. API response is processed and the relevant information is returned to the user through speech output

## Error Handling
- Implement error handling for invalid voice commands, image capture failures, and API request failures
- Provide appropriate error messages and guidance to the user through speech output

## Configuration
- Store Google Vision API credentials securely in the `.env` file
- Configure camera settings (resolution, frame rate) based on the connected camera

## Deployment
- Deploy the FastAPI server on a hosting platform
- Ensure the server is accessible from the device running the camera and voice interaction components

## Testing
- Write unit tests for the FastAPI endpoints and image processing logic
- Perform integration tests with the Google Vision API
- Conduct user testing with visually impaired individuals to gather feedback and improve usability

## .env File
- Create a `.env` file in the project root directory
- Add the following line to store the Google API key:
        GOOGLE_API_KEY=your_google_api_key_here

- Ensure the `.env` file is included in the `.gitignore` file to prevent sensitive information from being committed to version control


## Model Versions
1. models/gemini-1.0-pro
2. models/gemini-1.0-pro-001
3. models/gemini-1.0-pro-latest
4. models/gemini-1.0-pro-vision-latest
5. models/gemini-pro
6. models/gemini-pro-vision