import argparse
from io import BytesIO
import os
import cv2
import time
import torch
import logging
import numpy as np
import tempfile
import shutil
from fastapi import FastAPI, File, Query, UploadFile, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from config import generation_config, safety_settings
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image
from imagehash import phash
from tqdm import tqdm
from pathlib import Path

from Detection.Utils import ResizePadding
from CameraLoader import CamLoader, CamLoader_Q
from DetectorLoader import TinyYOLOv3_onecls
from PoseEstimateLoader import SPPE_FastPose
from fn import draw_single
from Track.Tracker import Detection, Tracker
from ActionsEstLoader import TSSTG

app = FastAPI()

orgins = [
    "http://localhost",
    "http://localhost:3000",
]


stream_running = True

app.add_middleware(
    CORSMiddleware,
    allow_origins=orgins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)

# Load Model
model = genai.GenerativeModel(
    model_name="gemini-pro-vision",
    generation_config=generation_config,
    safety_settings=safety_settings,
)


def preproc(image, resize_fn):
    """preprocess function for CameraLoader."""
    image = resize_fn(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def kpt2bbox(kpt, ex=20):
    """Get bbox that hold on all of the keypoints (x,y)
    kpt: array of shape `(N, 2)`,
    ex: (int) expand bounding box,
    """
    return np.array(
        (
            kpt[:, 0].min() - ex,
            kpt[:, 1].min() - ex,
            kpt[:, 0].max() + ex,
            kpt[:, 1].max() + ex,
        )
    )


def print_section_times(sections, section_times, start_time):
    total_time = time.time() - start_time
    prev_time = start_time

    print("Time taken for each section and their percentage of total runtime:")
    for _, (section, section_time) in enumerate(zip(sections, section_times)):
        section_duration = section_time - prev_time
        percentage = (section_duration / total_time) * 100
        print(f"{section}: {section_duration:.2f}s, {percentage:.2f}%")
        prev_time = section_time


@app.post("/process_video_with_gemini")
async def process_video_with_gemini(file_contents: bytes = File(...)):
    start_time = time.time()
    section_times = []
    section_times.append(time.time())  # Mark end of initialization

    output_directory = "selected_frames"
    os.makedirs(output_directory, exist_ok=True)

    # Save the uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(file_contents)
        temp_file.flush()
        video_path = temp_file.name

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Could not open video file")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    selected_frames = []
    previous_hashes = []
    hash_threshold = 15  # Adjust this threshold as needed
    max_frames = 14  # Maximum number of frames to select

    section_times.append(time.time())  # Mark end of setup

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
            if len(selected_frames) < max_frames:
                selected_frames.append(img)
                previous_hashes.append(current_hash)

                # Saving the selected frame to the output directory
                frame_filename = os.path.join(
                    output_directory, f"frame_{frame_idx:04d}.png"
                )
                cv2.imwrite(frame_filename, img)

        # Break the loop if the maximum number of frames is reached
        if len(selected_frames) >= max_frames:
            break

    section_times.append(time.time())  # Mark end of data capture

    # Releasing the video capture object and deleting the temporary file
    cap.release()
    os.unlink(video_path)

    print(f"Total key frames selected: {len(selected_frames)}")

    image_parts = []
    for i in range(len(selected_frames)):
        image_data = cv2.imencode(".png", selected_frames[i])[1].tobytes()
        image_part = {"mime_type": "image/png", "data": image_data}
        image_parts.append(image_part)

    # Get the prompt from prompt.py
    from prompt import prompt_parts

    prompt_parts.extend(image_parts)
    prompt_parts.append("Description:")

    section_times.append(time.time())  # Mark end of data processing

    try:
        # Pass the video and prompt to Google AI Studio Gemini
        response = model.generate_content(prompt_parts)
    except ValueError as e:
        print("Error occurred while generating content:")
        print(str(e))
        response = None

    section_times.append(time.time())  # Mark end of model generation

    shutil.rmtree(output_directory)

    sections = [
        "Initialization",
        "Setup",
        "Data Capture",
        "Data Processing",
        "Model Generation",
    ]
    print_section_times(sections, section_times, start_time)

    return response.text if response else None


def process_video(video_path, output_video_path, return_type):
    device = "cpu"  # or 'cuda'

    # DETECTION MODEL.
    inp_dets = 384
    detect_model = TinyYOLOv3_onecls(inp_dets, device=device)

    # POSE MODEL.
    inp_pose = "224x160"
    inp_pose = inp_pose.split("x")
    inp_pose = (int(inp_pose[0]), int(inp_pose[1]))
    pose_model = SPPE_FastPose("resnet50", inp_pose[0], inp_pose[1], device=device)

    # Tracker.
    max_age = 30
    tracker = Tracker(max_age=max_age, n_init=3)

    # Actions Estimate.
    action_model = TSSTG()

    resize_fn = ResizePadding(inp_dets, inp_dets)

    cam_source = video_path
    if type(cam_source) is str and os.path.isfile(cam_source):
        # Use loader thread with Q for video file.
        cam = CamLoader_Q(
            cam_source, queue_size=1000, preprocess=lambda x: preproc(x, resize_fn)
        ).start()
    else:
        # Use normal thread loader for webcam.
        cam = CamLoader(
            int(cam_source) if cam_source.isdigit() else cam_source,
            preprocess=lambda x: preproc(x, resize_fn),
        ).start()

    if return_type == "annotated_video":
        codec = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(output_video_path, codec, 30, (inp_dets, inp_dets))
    else:
        writer = None

    fps_time = 0
    f = 0
    detected_falls = []
    while cam.grabbed():
        f += 1
        frame = cam.getitem()
        image = frame.copy()

        # Detect humans bbox in the frame with detector model.
        detected = detect_model.detect(frame, need_resize=False, expand_bb=10)

        # Predict each tracks bbox of current frame from previous frames information with Kalman filter.
        tracker.predict()
        # Merge two source of predicted bbox together.
        for track in tracker.tracks:
            det = torch.tensor(
                [track.to_tlbr().tolist() + [0.5, 1.0, 0.0]], dtype=torch.float32
            )
            detected = (
                torch.cat([detected, det], dim=0) if detected is not None else det
            )

        detections = []  # List of Detections object for tracking.
        if detected is not None:
            # Predict skeleton pose of each bboxs.
            poses = pose_model.predict(frame, detected[:, 0:4], detected[:, 4])

            # Create Detections object.
            detections = [
                Detection(
                    kpt2bbox(ps["keypoints"].numpy()),
                    np.concatenate(
                        (ps["keypoints"].numpy(), ps["kp_score"].numpy()), axis=1
                    ),
                    ps["kp_score"].mean().numpy(),
                )
                for ps in poses
            ]

        # Update tracks by matching each track information of current and previous frame or
        # create a new track if no matched.
        tracker.update(detections)

        # Predict Actions of each track.
        for i, track in enumerate(tracker.tracks):
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            bbox = track.to_tlbr().astype(int)
            center = track.get_center().astype(int)

            action = "pending.."
            clr = (0, 255, 0)
            # Use 30 frames time-steps to prediction.
            if len(track.keypoints_list) == 30:
                pts = np.array(track.keypoints_list, dtype=np.float32)
                out = action_model.predict(pts, frame.shape[:2])
                action_name = action_model.class_names[out[0].argmax()]
                action = "{}: {:.2f}%".format(action_name, out[0].max() * 100)

                if action_name == "Fall Down":
                    clr = (255, 0, 0)
                    detected_falls.append(f)
                elif action_name == "Lying Down":
                    clr = (255, 200, 0)

            # VISUALIZE.
            if track.time_since_update == 0:
                if return_type == "annotated_video":
                    if True:  # Set to True to show skeleton
                        frame = draw_single(frame, track.keypoints_list[-1])
                    frame = cv2.rectangle(
                        frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1
                    )
                    frame = cv2.putText(
                        frame,
                        str(track_id),
                        (center[0], center[1]),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.4,
                        (255, 0, 0),
                        2,
                    )
                    frame = cv2.putText(
                        frame,
                        action,
                        (bbox[0] + 5, bbox[1] + 15),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.4,
                        clr,
                        1,
                    )

        if return_type == "annotated_video":
            frame = cv2.resize(frame, (inp_dets, inp_dets))
            writer.write(frame)

    # Clear resource.
    cam.stop()
    if writer is not None:
        writer.release()

    if return_type == "fall_detection":
        return {"detected_falls": detected_falls}
    else:
        return None


def process_stream(
    cam_source,
    inp_dets,
    inp_pose,
    device,
    show_detected,
    show_skeleton,
    return_type,
    timeout,
):
    resize_fn = ResizePadding(inp_dets, inp_dets)

    if type(cam_source) is str and os.path.isfile(cam_source):
        # Use loader thread with Q for video file.
        cam = CamLoader_Q(
            cam_source, queue_size=1000, preprocess=lambda x: preproc(x, resize_fn)
        ).start()
    else:
        # Use normal thread loader for webcam.
        cam = CamLoader(
            cam_source if isinstance(cam_source, str) else int(cam_source),
            preprocess=lambda x: preproc(x, resize_fn),
        ).start()

    try:
        # DETECTION MODEL.
        detect_model = TinyYOLOv3_onecls(inp_dets, device=device)

        # POSE MODEL.
        pose_model = SPPE_FastPose("resnet50", inp_pose[0], inp_pose[1], device=device)

        # Tracker.
        max_age = 30
        tracker = Tracker(max_age=max_age, n_init=3)

        # Actions Estimate.
        action_model = TSSTG()

        fps_time = 0
        f = 0
        start_time = time.time()
        logs = []
        while cam.grabbed() and time.time() - start_time < timeout and stream_running:
            f += 1
            frame = cam.getitem()
            image = frame.copy()

            # Detect humans bbox in the frame with detector model.
            detected = detect_model.detect(frame, need_resize=False, expand_bb=10)

            # Predict each tracks bbox of current frame from previous frames information with Kalman filter.
            tracker.predict()
            # Merge two source of predicted bbox together.
            for track in tracker.tracks:
                det = torch.tensor(
                    [track.to_tlbr().tolist() + [0.5, 1.0, 0.0]], dtype=torch.float32
                )
                detected = (
                    torch.cat([detected, det], dim=0) if detected is not None else det
                )

            detections = []  # List of Detections object for tracking.
            if detected is not None:
                # Predict skeleton pose of each bboxs.
                poses = pose_model.predict(frame, detected[:, 0:4], detected[:, 4])

                # Create Detections object.
                detections = [
                    Detection(
                        kpt2bbox(ps["keypoints"].numpy()),
                        np.concatenate(
                            (ps["keypoints"].numpy(), ps["kp_score"].numpy()), axis=1
                        ),
                        ps["kp_score"].mean().numpy(),
                    )
                    for ps in poses
                ]

                # VISUALIZE.
                if show_detected:
                    for bb in detected[:, 0:5]:
                        x1 = int(bb[0].item())
                        y1 = int(bb[1].item())
                        x2 = int(bb[2].item())
                        y2 = int(bb[3].item())
                        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)

            # Update tracks by matching each track information of current and previous frame or
            # create a new track if no matched.
            tracker.update(detections)

            # Predict Actions of each track.
            for i, track in enumerate(tracker.tracks):
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                bbox = track.to_tlbr().astype(int)
                center = track.get_center().astype(int)

                action = "pending.."
                clr = (0, 255, 0)
                # Use 30 frames time-steps to prediction.
                if len(track.keypoints_list) == 30:
                    pts = np.array(track.keypoints_list, dtype=np.float32)
                    out = action_model.predict(pts, frame.shape[:2])
                    action_name = action_model.class_names[out[0].argmax()]
                    action = "{}: {:.2f}%".format(action_name, out[0].max() * 100)

                    # Log the user event
                    log_message = f"User {track_id}: {action_name}"
                    logging.info(log_message)
                    logs.append(log_message)

                    if action_name == "Fall Down":
                        clr = (255, 0, 0)
                    elif action_name == "Lying Down":
                        clr = (255, 200, 0)

                # VISUALIZE.
                if track.time_since_update == 0:
                    if show_skeleton:
                        frame = draw_single(frame, track.keypoints_list[-1])
                    frame = cv2.rectangle(
                        frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1
                    )
                    frame = cv2.putText(
                        frame,
                        str(track_id),
                        (center[0], center[1]),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.4,
                        (255, 0, 0),
                        2,
                    )
                    frame = cv2.putText(
                        frame,
                        action,
                        (bbox[0] + 5, bbox[1] + 15),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.4,
                        clr,
                        1,
                    )

            # Show Frame.
            frame = cv2.resize(frame, (0, 0), fx=2.0, fy=2.0)
            frame = cv2.putText(
                frame,
                "%d, FPS: %f" % (f, 1.0 / (time.time() - fps_time)),
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
            frame = frame[:, :, ::-1]
            fps_time = time.time()

            if return_type == "annotated_stream":
                # Convert frame to JPEG format
                _, encoded_frame = cv2.imencode(".jpg", frame)
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + encoded_frame.tobytes()
                    + b"\r\n"
                )
            elif return_type == "logs":
                log_text = "\n".join(logs)
                yield log_text

    finally:
        # Clear resource.
        cam.stop()

def create_video_from_frames(frames, output_path, fps=30):
    if len(frames) == 0:
        return None

    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        video_writer.write(frame)

    video_writer.release()
    return output_path

@app.get("/stream")
async def stream(request: Request):
    print("Received request for /stream endpoint")

    cam_source = int(request.query_params.get("camera", 0))
    print(f"Camera source: {cam_source}")

    inp_dets = int(request.query_params.get("detection_input_size", 384))
    print(f"Detection input size: {inp_dets}")

    inp_pose = request.query_params.get("pose_input_size", "224x160")
    print(f"Pose input size (raw): {inp_pose}")
    inp_pose = inp_pose.split("x")
    inp_pose = (int(inp_pose[0]), int(inp_pose[1]))
    print(f"Pose input size (parsed): {inp_pose}")

    device = request.query_params.get("device", "cpu")
    print(f"Device: {device}")

    show_detected = request.query_params.get("show_detected", "False").lower() == "true"
    print(f"Show detected: {show_detected}")

    show_skeleton = request.query_params.get("show_skeleton", "True").lower() == "true"
    print(f"Show skeleton: {show_skeleton}")

    return_type = request.query_params.get("return_type", "annotated_stream")
    print(f"Return type: {return_type}")

    timeout = int(request.query_params.get("timeout", 15))
    print(f"Timeout: {timeout} seconds")

    print("Starting to process the stream...")
    stream_generator = process_stream(
        cam_source,
        inp_dets,
        inp_pose,
        device,
        show_detected,
        show_skeleton,
        return_type,
        timeout,
    )

    if return_type == "annotated_stream":
        return StreamingResponse(
            stream_generator,
            media_type="multipart/x-mixed-replace;boundary=frame",
        )
    elif return_type == "logs":
        return StreamingResponse(stream_generator, media_type="text/plain")

@app.post("/trace_video")
async def trace_video(
    file: UploadFile = File(...),
    return_type: str = Query("annotated_video", description="Return type for the processed video")
):
    # Get the current working directory
    current_dir = os.getcwd()

    # Create a file path to store the uploaded video in the current working directory
    video_path = os.path.join(current_dir, "uploaded_video.mp4")

    # Save the uploaded video to the file path
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if return_type == "annotated_video":
        # Create a file path to store the output video in the current working directory
        output_path = os.path.join(current_dir, "traced_video.mp4")
        # Process the video and generate the traced output video file
        process_video(video_path, output_path, return_type)
        # Return the traced video file as a response
        return FileResponse(
            output_path, media_type="video/mp4", filename="traced_video.mp4"
        )
    elif return_type == "fall_detection":
        # Process the video and return the detected falls
        result = process_video(video_path, None, return_type)
        detected_falls = result["detected_falls"]

        # Load the video file
        video = cv2.VideoCapture(video_path)

        # Extract frames where falls are detected
        fall_frames = []
        frame_count = 0
        while True:
            ret, frame = video.read()
            if not ret:
                break
            if frame_count in detected_falls:
                fall_frames.append(frame)
            frame_count += 1

        video.release()

        # Create a temporary file for the fall detection video
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            fall_video_path = temp_file.name

        # Create a video from the fall frames
        fall_video_path = create_video_from_frames(fall_frames, fall_video_path)

    if fall_video_path:
        # Open the fall detection video file
        with open(fall_video_path, "rb") as file:
            # Create an UploadFile object from the file
            file_contents = file.read()

        
        # Pass the fall detection video to the process_with_gemini function
        response = await process_video_with_gemini(file_contents)
        return response
    else:
        return "No falls detected in the video."


@app.post("/stop_stream")
async def stop_stream():
    global stream_running
    stream_running = False
    return {"message": "Stream stopped"}

if __name__ == "__main__":
    import uvicorn
    import argparse
    import socket

    par = argparse.ArgumentParser(description="Human Fall Detection Demo.")
    par.add_argument(
        "-C", "--camera", default=0, help="Source of camera or video file path."
    )
    par.add_argument(
        "--detection_input_size",
        type=int,
        default=384,
        help="Size of input in detection model in square must be divisible by 32 (int).",
    )
    par.add_argument(
        "--pose_input_size",
        type=str,
        default="224x160",
        help="Size of input in pose model must be divisible by 32 (h, w)",
    )
    par.add_argument(
        "--pose_backbone",
        type=str,
        default="resnet50",
        help="Backbone model for SPPE FastPose model.",
    )
    par.add_argument(
        "--show_detected",
        default=False,
        action="store_true",
        help="Show all bounding box from detection.",
    )
    par.add_argument(
        "--show_skeleton", default=True, action="store_true", help="Show skeleton pose."
    )
    par.add_argument(
        "--save_out", type=str, default="", help="Save display to video file."
    )
    par.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run model on cpu or cuda.",
    )
    args = par.parse_args()

    # Get the local IP address
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    # Run the server on the local IP address
    uvicorn.run(app, host=local_ip, port=8000)
