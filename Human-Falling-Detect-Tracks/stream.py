import argparse
import cv2
import numpy as np
import requests

def stream_response(url, camera, detection_input_size, pose_input_size, device, show_detected, show_skeleton, return_type, timeout):
    params = {
        "camera": camera,
        "detection_input_size": detection_input_size,
        "pose_input_size": pose_input_size,
        "device": device,
        "show_detected": show_detected,
        "show_skeleton": show_skeleton,
        "return_type": return_type,
        "timeout": timeout,
    }
    try:
        print("Sending GET request to the server...")
        response = requests.get(url, params=params, stream=True)
        print(f"Response status code: {response.status_code}")
        response.raise_for_status()

        if return_type == "annotated_stream":
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
        elif return_type == "logs":
            print("Receiving logs from the server...")
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    print(line)

    except requests.exceptions.RequestException as e:
        print("Error:", e)
    finally:
        if return_type == "annotated_stream":
            print("Closing the window...")
            cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream response from a server.")
    parser.add_argument("--url", type=str, default="http://localhost:8000/stream", help="URL of the server")
    parser.add_argument("--camera", type=str, default="0", help="Camera ID or path")
    parser.add_argument("--detection_input_size", type=str, default="384", help="Detection input size")
    parser.add_argument("--pose_input_size", type=str, default="224x160", help="Pose input size")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (e.g., cpu, cuda)")
    parser.add_argument("--show_detected", type=str, default="False", help="Show detected objects")
    parser.add_argument("--show_skeleton", type=str, default="True", help="Show skeleton")
    parser.add_argument("--return_type", type=str, default="logs", help="Return type (annotated_stream or logs)")
    parser.add_argument("--timeout", type=int, default=15, help="Timeout duration in seconds")
    args = parser.parse_args()

    stream_response(
        url=args.url,
        camera=args.camera,
        detection_input_size=args.detection_input_size,
        pose_input_size=args.pose_input_size,
        device=args.device,
        show_detected=args.show_detected,
        show_skeleton=args.show_skeleton,
        return_type=args.return_type,
        timeout=args.timeout,
    )