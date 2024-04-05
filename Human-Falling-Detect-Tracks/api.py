import os
import cv2
import time
import torch
import logging
import numpy as np
import tempfile
import shutil
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse

from Detection.Utils import ResizePadding
from CameraLoader import CamLoader, CamLoader_Q
from DetectorLoader import TinyYOLOv3_onecls
from PoseEstimateLoader import SPPE_FastPose
from fn import draw_single
from Track.Tracker import Detection, Tracker
from ActionsEstLoader import TSSTG

app = FastAPI()


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


def process_video(video_path, output_video_path):
    device = "cpu"  # or 'cpu'

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

    codec = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, codec, 30, (inp_dets * 2, inp_dets * 2))

    fps_time = 0
    f = 0
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

            # VISUALIZE.
            if True:
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
                logging.info(f"User {track_id}: {action_name}")

                if action_name == "Fall Down":
                    clr = (255, 0, 0)
                elif action_name == "Lying Down":
                    clr = (255, 200, 0)

            # VISUALIZE.
            if track.time_since_update == 0:
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

        writer.write(frame)

    # Clear resource.
    cam.stop()
    writer.release()


@app.post("/trace_video")
async def trace_video(file: UploadFile = File(...)):
    # Get the current working directory
    current_dir = os.getcwd()

    # Create a file path to store the uploaded video in the current working directory
    video_path = os.path.join(current_dir, "uploaded_video.mp4")

    # Save the uploaded video to the file path
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Create a file path to store the output video in the current working directory
    output_path = os.path.join(current_dir, "traced_video.mp4")

    # Process the video and generate the traced output video file
    process_video(video_path, output_path)

    # Return the traced video file as a response
    return FileResponse(output_path, media_type="video/mp4", filename="traced_video.mp4")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
