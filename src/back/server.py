import argparse
import asyncio
import json
import logging
import os
import ssl
import uuid
import numpy as np

import cv2
from aiohttp import web
from aiortc import (
    MediaStreamTrack,
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack,
)
from aiortc.contrib.media import MediaRelay
from av import VideoFrame

import mediapipe as mp

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()
relay = MediaRelay()
consumer_track = VideoStreamTrack()
effects = dict()

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    smooth_landmarks=True,
)

segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)

# Drawing utility
mp_drawing = mp.solutions.drawing_utils


class VideoTransformTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from another track.
    """

    kind = "video"

    def __init__(self, track, transform):
        super().__init__()  # don't forget this!
        self.track = track
        self.transform = transform

    async def recv(self):
        frame = await self.track.recv()

        if self.transform == "cartoon":
            img = frame.to_ndarray(format="bgr24")

            # prepare color
            img_color = cv2.pyrDown(cv2.pyrDown(img))
            for _ in range(6):
                img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
            img_color = cv2.pyrUp(cv2.pyrUp(img_color))

            # prepare edges
            img_edges = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img_edges = cv2.adaptiveThreshold(
                cv2.medianBlur(img_edges, 7),
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                9,
                2,
            )
            img_edges = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2RGB)

            # combine color and edges
            img = cv2.bitwise_and(img_color, img_edges)

            # rebuild a VideoFrame, preserving timing information
            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame
        elif self.transform == "edges":
            # perform edge detection
            img = frame.to_ndarray(format="bgr24")
            img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)

            # rebuild a VideoFrame, preserving timing information
            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame
        elif self.transform == "rotate":
            # rotate image
            img = frame.to_ndarray(format="bgr24")
            rows, cols, _ = img.shape
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), frame.time * 45, 1)
            img = cv2.warpAffine(img, M, (cols, rows))

            # rebuild a VideoFrame, preserving timing information
            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame
        elif self.transform == "skeleton":
            # Convert the aiortc frame to an array
            img = frame.to_ndarray(format="bgr24")

            # Process the frame for skeleton
            img = await process_frame_for_skeleton(img)

            # Rebuild a VideoFrame, preserving timing information
            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame
        
        elif self.transform == "segmentation":
            # Convert the aiortc frame to an array
            img = frame.to_ndarray(format="bgr24")

            # Process the frame for skeleton
            # img = await process_frame_for_skeleton(img)
            img = await process_frame_for_segmentation(img)

            # Rebuild a VideoFrame, preserving timing information
            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame

        else:
            return frame


async def consumer(request):
    if request.method == "OPTIONS":
        return web.Response(
            content_type="application/json",
            headers={"Access-Control-Allow-Origin": "*",
                     "Access-Control-Allow-Credentials": "true",
                     "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
                     "Access-Control-Allow-Headers": "Content-Type"},
        )

    params = await request.json()
    annotation = params["video_transform"]

    description = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Track %s sent", consumer_track.kind)
    pc.addTrack(
        relay.subscribe(effects[annotation])
    )

    await pc.setRemoteDescription(description)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
        headers={"Access-Control-Allow-Origin": "*",
                 "Access-Control-Allow-Credentials": "true",
                 "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
                 "Access-Control-Allow-Headers": "Content-Type"},
    )


async def broadcast(request):
    if request.method == "OPTIONS":
        return web.Response(
            content_type="application/json",
            headers={"Access-Control-Allow-Origin": "*",
                     "Access-Control-Allow-Credentials": "true",
                     "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
                     "Access-Control-Allow-Headers": "Content-Type"},
        )

    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            if isinstance(message, str) and message.startswith("ping"):
                channel.send("pong" + message[4:])

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        log_info("Connection state is %s", pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        log_info("Track %s received", track.kind)

        if track.kind == "video":
            global consumer_track
            consumer_track = track
            effects["source"] = relay.subscribe(track)
            effects["skeleton"] = VideoTransformTrack(relay.subscribe(track), "skeleton")
            effects["cartoon"] = VideoTransformTrack(relay.subscribe(track), "cartoon")
            effects["edges"] = VideoTransformTrack(relay.subscribe(track), "edges")
            effects["segmentation"] = VideoTransformTrack(relay.subscribe(track), "segmentation")

        @track.on("ended")
        async def on_ended():
            log_info("Track %s ended", track.kind)

    # handle offer
    await pc.setRemoteDescription(offer)

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
        headers={"Access-Control-Allow-Origin": "*",
                 "Access-Control-Allow-Credentials": "true",
                 "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
                 "Access-Control-Allow-Headers": "Content-Type"},
    )


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


async def process_frame_for_skeleton(frame):
    # Convert the BGR frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame
    results = pose.process(frame_rgb)

    # Draw the pose annotations on the frame
    annotated_frame = frame.copy()
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            annotated_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

    return annotated_frame

async def process_frame_for_segmentation(frame):
    # Convert the BGR frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame
    results = segmentation.process(frame_rgb)
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.95
    ouput_frame = np.where(condition, frame, 255).astype(np.uint8)

    return ouput_frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WebRTC audio / video / data-channels demo"
    )
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_post("/broadcast", broadcast)
    app.router.add_options("/broadcast", broadcast)
    app.router.add_post("/consumer", consumer)
    app.router.add_options("/consumer", consumer)
    web.run_app(
        app, access_log=None, host=args.host, port=args.port, ssl_context=ssl_context
    )
