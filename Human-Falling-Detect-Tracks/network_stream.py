import argparse
import base64
import requests
from fastapi import FastAPI, Request, Response
import uvicorn
import cv2
import numpy as np

def generate_html(url, camera, detection_input_size, pose_input_size, device, show_detected, show_skeleton):
    html = f"""
    <html>
    <head>
        <title>Video Stream</title>
    </head>
    <body>
        <img id="video" src="" />
        <script>
            var url = "{url}";
            var params = {{
                "camera": "{camera}",
                "detection_input_size": "{detection_input_size}",
                "pose_input_size": "{pose_input_size}",
                "device": "{device}",
                "show_detected": "{show_detected}",
                "show_skeleton": "{show_skeleton}"
            }};
            var img = document.getElementById("video");
            function updateImage() {{
                fetch(url, {{ method: "POST", body: JSON.stringify(params) }})
                    .then(response => response.json())
                    .then(data => {{
                        img.src = "data:image/jpeg;base64," + data.frame;
                        setTimeout(updateImage, 30);
                    }});
            }}
            updateImage();
        </script>
    </body>
    </html>
    """
    return html

app = FastAPI()

@app.get("/")
def get_html(request: Request):
    html = generate_html(
        request.url_for("get_frame"),
        request.query_params.get("camera", "0"),
        request.query_params.get("detection_input_size", "384"),
        request.query_params.get("pose_input_size", "224x160"),
        request.query_params.get("device", "cpu"),
        request.query_params.get("show_detected", "True"),
        request.query_params.get("show_skeleton", "True")
    )
    return Response(content=html, media_type="text/html")

@app.post("/frame")
async def get_frame(request: Request):
    params = await request.json()
    response = requests.get(request.query_params.get("url", "http://localhost:8000/stream"), params=params, stream=True)
    
    jpg = b""
    for chunk in response.iter_content(chunk_size=1024):
        jpg += chunk
        a = jpg.find(b'\xff\xd8')
        b = jpg.find(b'\xff\xd9')
        if a != -1 and b != -1:
            frame_data = jpg[a:b+2]
            jpg = jpg[b+2:]
            frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                _, encoded_frame = cv2.imencode('.jpg', frame)
                frame_base64 = base64.b64encode(encoded_frame).decode('utf-8')
                return {"frame": frame_base64}
    
    return {"frame": ""}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream response from a server.")
    parser.add_argument("--url", type=str, default="http://localhost:8000/stream", help="URL of the server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8080, help="Port to run the server on")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)