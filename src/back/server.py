import argparse
import asyncio
import json
import logging
import os
import ssl
import uuid

from aiohttp import web
from aiortc import (
    MediaStreamTrack,
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack,
)
from aiortc.contrib.media import MediaRelay
from jsonschema import validate, ValidationError

from videotransformtrack import VideoTransformTrack, UnsupportedTransform

RESPONSE_HEADER = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Credentials": "true",
    "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type"
}

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
peer_connections = set()
relay = MediaRelay()
source_video = VideoStreamTrack()
broadcaster_active = False
NO_TRANSFORM: str = "none"
_transformed_tracks = dict()


def apply_transform(track: MediaStreamTrack, transform: str) -> MediaStreamTrack:
    if transform == NO_TRANSFORM:
        return track
    if transform not in _transformed_tracks:
        _transformed_tracks[transform] = VideoTransformTrack(track, transform)
    return _transformed_tracks[transform]


consumer_schema = {
    "type": "object",
    "properties": {
        "sdp": {"type": "string"},
        "type": {"type": "string"},
        "video_transform": {"type": "string"},
    },
    "required": ["sdp", "type", "video_transform"],
}

broadcast_schema = {
    "type": "object",
    "properties": {
        "sdp": {"type": "string"},
        "type": {"type": "string"},
    },
    "required": ["sdp", "type"],
}


async def consume(request):
    if request.method == "OPTIONS":
        return web.Response(
            content_type="application/json",
            headers=RESPONSE_HEADER,
        )

    body = await request.json()
    try:
        validate(instance=body, schema=consumer_schema)
    except ValidationError as error:
        raise web.HTTPBadRequest(reason=error.message)

    description = RTCSessionDescription(sdp=body["sdp"], type=body["type"])
    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    peer_connections.add(pc)

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    video_transform = NO_TRANSFORM if not body["video_transform"] else body["video_transform"].lower()
    try:
        transformed_track = apply_transform(source_video, transform=video_transform)
    except UnsupportedTransform as error:
        raise web.HTTPBadRequest(reason=error.message)

    pc.addTrack(relay.subscribe(transformed_track))
    log_info("Track %s sent", source_video.kind)

    await pc.setRemoteDescription(description)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps({
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        }),
        headers=RESPONSE_HEADER,
    )


async def broadcast(request):
    if request.method == "OPTIONS":
        return web.Response(
            content_type="application/json",
            headers=RESPONSE_HEADER,
        )

    body = await request.json()
    try:
        validate(instance=body, schema=broadcast_schema)
    except ValidationError as error:
        raise web.HTTPBadRequest(reason=error.message)

    # global broadcaster_active
    # if broadcaster_active:
    #     raise web.HTTPForbidden(reason="Another broadcaster is already active.")

    offer = RTCSessionDescription(sdp=body["sdp"], type=body["type"])

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    peer_connections.add(pc)

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
            peer_connections.discard(pc)

    @pc.on("track")
    def on_track(track):
        log_info("Track %s received", track.kind)

        if track.kind == "video":
            global source_video, broadcaster_active
            source_video = track
            broadcaster_active = True

        @track.on("ended")
        async def on_ended():
            log_info("Track %s ended", track.kind)
            _transformed_tracks.clear()
            global broadcaster_active
            broadcaster_active = False
            print("broadcast ended")

    # handle offer
    await pc.setRemoteDescription(offer)

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps({
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        }),
        headers=RESPONSE_HEADER,
    )


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in peer_connections]
    await asyncio.gather(*coros)
    peer_connections.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebRTC audio / video / data-channels demo")
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument("--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="Port for HTTP server (default: 8080)")
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
    app.router.add_post("/consume", consume)
    app.router.add_options("/consume", consume)
    web.run_app(app, access_log=None, host=args.host, port=args.port, ssl_context=ssl_context)
