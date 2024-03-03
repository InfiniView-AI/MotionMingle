from typing import Dict, Callable

import cv2
import mediapipe as mp
from aiortc import MediaStreamTrack
from av import VideoFrame
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    smooth_landmarks=True,
)

segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)
mp_drawing = mp.solutions.drawing_utils


class VideoTransformTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from another track.
    """

    kind = "video"

    def __init__(self, track: MediaStreamTrack, transform: str):
        if transform not in VideoTransformTrack.__TRANSFORMERS:
            raise UnsupportedTransform(f"transform \"{transform}\" is not supported")
        super().__init__()  # don't forget this!
        self.track = track
        self.transform = transform

    async def recv(self) -> VideoFrame:
        frame = await self.track.recv()
        return VideoTransformTrack.__TRANSFORMERS[self.transform](frame)

    @staticmethod
    def _cartoon_transformer(frame: VideoFrame) -> VideoFrame:
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

    @staticmethod
    def _edges_transformer(frame: VideoFrame) -> VideoFrame:
        # perform edge detection
        img = frame.to_ndarray(format="bgr24")
        img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)

        # rebuild a VideoFrame, preserving timing information
        new_frame = VideoFrame.from_ndarray(img, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame

    @staticmethod
    def _rotate_transformer(frame: VideoFrame) -> VideoFrame:
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

    @staticmethod
    def _skeleton_transformer(frame: VideoFrame) -> VideoFrame:
        # Convert the aiortc frame to an array
        img = frame.to_ndarray(format="bgr24")

        # Convert the BGR frame to RGB
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the frame
        results = pose.process(frame_rgb)

        # Draw the pose annotations on the frame
        annotated_frame = img.copy()
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(annotated_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Rebuild a VideoFrame, preserving timing information
        new_frame = VideoFrame.from_ndarray(annotated_frame, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame
    


    @staticmethod
    def _segmentation_transformer(frame: VideoFrame) -> VideoFrame:
        # Convert the aiortc frame to an array

        def process_frame_for_segmentation(frame):
            # Convert the BGR frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame
            results = segmentation.process(frame_rgb)
            condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.95
            ouput_frame = np.where(condition, frame, frame_rgb*0.5).astype(np.uint8)

            return ouput_frame
        
        img = frame.to_ndarray(format="bgr24")
        
        annotated_frame = process_frame_for_segmentation(img)

        # Rebuild a VideoFrame, preserving timing information
        new_frame = VideoFrame.from_ndarray(annotated_frame, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame

    __TRANSFORMERS: Dict[str, Callable[[VideoFrame], VideoFrame]] = {
        "cartoon": _cartoon_transformer,
        "edges": _edges_transformer,
        "rotate": _rotate_transformer,
        "skeleton": _skeleton_transformer,
        "segmentation": _segmentation_transformer,
    }

    supported_transforms = list(__TRANSFORMERS.keys())


class UnsupportedTransform(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
