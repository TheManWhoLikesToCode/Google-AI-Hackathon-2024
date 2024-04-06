import cv2
import numpy as np
import requests

def stream_response():
    url = "http://localhost:8000/stream"
    params = {
        "camera": "0",
        "detection_input_size": "384",
        "pose_input_size": "224x160",
        "device": "cpu",
        "show_detected": "False",
        "show_skeleton": "True",
    }

    try:
        print("Sending GET request to the server...")
        response = requests.get(url, params=params, stream=True)
        print(f"Response status code: {response.status_code}")
        response.raise_for_status()

        print("Creating a window to display the video stream...")
        cv2.namedWindow("Stream", cv2.WINDOW_NORMAL)

        jpg = b""
        frame_count = 0

        for chunk in response.iter_content(chunk_size=1024):
            jpg += chunk
            a = jpg.find(b'\xff\xd8')
            b = jpg.find(b'\xff\xd9')
            if a != -1 and b != -1:
                frame_data = jpg[a:b+2]
                jpg = jpg[b+2:]

                frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    print(f"Displaying frame {frame_count} with shape: {frame.shape}...")
                    cv2.imshow("Stream", frame)
                    frame_count += 1
                    if cv2.waitKey(1) == ord("q"):
                        print("User pressed 'q' key. Exiting...")
                        break
                else:
                    print("Failed to decode JPEG frame.")

    except requests.exceptions.RequestException as e:
        print("Error:", e)

    finally:
        print("Closing the window...")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    stream_response()