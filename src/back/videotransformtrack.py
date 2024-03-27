from typing import Dict, Callable, List

import cv2
import mediapipe as mp
from aiortc import MediaStreamTrack
from av import VideoFrame
import numpy as np
import COM

from mediapipe.framework.formats import landmark_pb2
import math
from BSP import BSP

solutions = mp.solutions

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    smooth_landmarks=True,
)

segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)
mp_drawing = mp.solutions.drawing_utils

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode



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
            condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.4
            ouput_frame = np.where(condition, frame, frame_rgb * 0.4).astype(np.uint8)

            return ouput_frame
        
        img = frame.to_ndarray(format="bgr24")
        
        annotated_frame = process_frame_for_segmentation(img)

        # Rebuild a VideoFrame, preserving timing information
        new_frame = VideoFrame.from_ndarray(annotated_frame, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame
    
    @staticmethod
    def _center_of_mass_transformer(frame: VideoFrame) -> VideoFrame:
        VideoTransformTrack.timestamp += 1
        img = frame.to_ndarray(format="bgr24")

        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

        VideoTransformTrack.landmarker.detect_async(img, VideoTransformTrack.timestamp)
        if (not(VideoTransformTrack.COM_result is None)):
            new_frame = VideoTransformTrack.__draw_landmarks_on_image(img.numpy_view(), VideoTransformTrack.COM_result)
            new_frame = VideoFrame.from_ndarray(new_frame, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame

        return frame

    @staticmethod
    def __draw_landmarks_on_image(rgb_image: np.ndarray, detection_result) -> np.ndarray:
        pose_landmarks_list = detection_result.pose_landmarks
        annotated_image = np.copy(rgb_image)
        
        cv2.rectangle(annotated_image,(0,0),(200,200),(255,255,255),-1)

        # Loop through the detected poses to visualize.
        for idx in range(len(pose_landmarks_list)):
            # length is 33
            # check this guide to see what each index represents: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
            pose_landmarks = pose_landmarks_list[idx]

            # Draw the pose landmarks.
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())


            VideoTransformTrack.__draw_COM(annotated_image, pose_landmarks, BSP)


        return annotated_image

    @staticmethod
    def __set_COM_result(result: PoseLandmarkerResult, *args) -> None:
        VideoTransformTrack.COM_result = result

    
    @staticmethod
    def __draw_COM(img: np.ndarray, pose_landmarks: List[landmark_pb2.NormalizedLandmark], BSP: Dict[str, float]):
        SCREEN_SCALE = 200
        PIC_SCALE = 40
        LIFTED_THRESHOLD = 0.025
        COM_DISTANCE_THRESHOLD = 0.7
        
        def average_position(pose_landmarks, indices):
            """Calculate the average position of specified landmarks."""
            if isinstance(indices, tuple):
                x = sum(pose_landmarks[i].x for i in indices) / len(indices)
                y = sum(pose_landmarks[i].y for i in indices) / len(indices)
                z = sum(pose_landmarks[i].z for i in indices) / len(indices)
                return x, y, z
            else:
                return pose_landmarks[indices].x, pose_landmarks[indices].y, pose_landmarks[indices].z

        def interpolate_segment(origin, other, l):
            """Interpolate a segment between two points."""
            return (np.interp(l, [0, 1], [origin[i], other[i]]) for i in range(3))

        def get_segment_mass(segment, pose_landmarks, BSP):
            """Calculate the mass for a body segment."""
            origin = average_position(pose_landmarks, BSP[segment]['origin'])
            other = average_position(pose_landmarks, BSP[segment]['other'])
            segment_position = interpolate_segment(origin, other, BSP[segment]['l'])
            return tuple(pos * BSP[segment]['m'] for pos in segment_position)

        def get_COM(pose_landmarks, BSP, SCREEN_SCALE):
            """Calculate the Center of Mass (COM) for the pose."""
            com_x, com_y, com_z = 0.0, 0.0, 0.0
            for segment in BSP:
                segment_mass = get_segment_mass(segment, pose_landmarks, BSP)
                com_x += segment_mass[0]
                com_y += segment_mass[1]
                com_z += segment_mass[2]
            return com_x * SCREEN_SCALE, com_y * SCREEN_SCALE, com_z * SCREEN_SCALE
            
        def rotate_image(image, angle):
            """Rotate an image by a specific angle."""
            row, col = image.shape[:2]
            center = tuple(np.array([row, col]) / 2)
            rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            return cv2.warpAffine(image, rot_mat, (col, row))

        def get_foot_angle(foot_heel, foot_toe):
            """Calculate the angle for a foot."""
            return math.degrees(math.atan2(foot_heel.z - foot_toe.z, foot_heel.x - foot_toe.x)) + 90

        
        def get_foot_to_com_distance(x, z, com):
            """Calculate the distance from a foot to the COM."""
            return abs((x - com[0]) ** 2 + (z - com[2]) ** 2)
    
        def set_footprint_colour(footprint, colour):
            """Map colour names to BGR values"""
            colour_map = {
                "blue": (255, 255, 0),
                "orange": (0, 165, 255),
                "red": (0, 0, 255),
                "dark_red": (0, 0, 122),
            }

            # Check if the colour is in the map
            if colour in colour_map:
                b, g, r = colour_map[colour]
                footprint[:, :, 0] = b
                footprint[:, :, 1] = g
                footprint[:, :, 2] = r
            else:
                print("Colour not recognized. Please add it to the colour_map.")

            return footprint

        def get_foot_to_com_distance(x, z, com):
            """Calculate the distance from a foot to the COM."""
            return abs((x - com[0]) **2 + (z - com[2]) ** 2)
        
        def check_dimensions(crop, pic_scale):
            """Check the if picture dimension matches"""
            return crop.shape[0] != pic_scale or crop.shape[1] != pic_scale

        def overlay_image(alpha_mask, alpha_inv, src_img, overlay_img, y1, y2, x1, x2, channel):
            """Overlay image to frame"""
            src_img[y1:y2, x1:x2, channel] = (alpha_mask * overlay_img[:, :, channel] +
                                            alpha_inv * src_img[y1:y2, x1:x2, channel])

        def extract_alpha_masks(image):
            """Extracts the alpha mask and its inverse from an image."""
            alpha_mask = image[:, :, 3] / 255.0
            alpha_inv = 1.0 - alpha_mask
            return alpha_mask, alpha_inv
        
        def calculate_positions(offset_x, offset_y, shape):
            """Calculate the y1, y2, x1, x2 positions based on offsets and image shape."""
            y1, y2 = offset_y, offset_y + shape[0]
            x1, x2 = offset_x, offset_x + shape[1]
            return y1, y2, x1, x2

        left_heel = pose_landmarks[29] # left heel
        left_toe = pose_landmarks[31] # left toe

        right_heel = pose_landmarks[30] # right heel
        right_toe = pose_landmarks[32] # right toe

        left_shoulder = pose_landmarks[11]
        right_shoulder = pose_landmarks[12]

        # angle
        left_foot_angle = get_foot_angle(left_heel, left_toe)
        right_foot_angle = get_foot_angle(right_heel, right_toe)
        shoulder_angle = get_foot_angle(left_shoulder, right_shoulder)
        # load pics
        right_footprint = cv2.imread("right_footprint.png", -1)
        left_footprint = cv2.imread("left_footprint.png", -1)
        face = cv2.resize(cv2.imread("face.png", -1), (PIC_SCALE, PIC_SCALE))
        face = rotate_image(face, shoulder_angle + 90)
        
        # resize, change colour and rotate
        right_footprint = cv2.resize(right_footprint, (PIC_SCALE, PIC_SCALE))
        right_footprint = set_footprint_colour(right_footprint, "blue")
        right_footprint = rotate_image(right_footprint, right_foot_angle)

        left_footprint = cv2.resize(left_footprint, (PIC_SCALE, PIC_SCALE))
        left_footprint = rotate_image(left_footprint, left_foot_angle)
        left_footprint = set_footprint_colour(left_footprint, "orange")

        # calculate offset
        right_foot_x_offset = math.ceil((right_heel.x + right_toe.x) / 2 * SCREEN_SCALE)
        right_foot_y_offset = -math.ceil((right_heel.z + right_toe.z) / 2 * SCREEN_SCALE) + int(SCREEN_SCALE / 2)
        right_foot_z_offset = math.ceil((right_heel.z + right_toe.z) / 2 * SCREEN_SCALE)

        left_foot_x_offset = math.ceil((left_heel.x + left_toe.x) / 2 * SCREEN_SCALE) # feet width adjustment
        left_foot_y_offset = -math.ceil((left_heel.z + left_toe.z) / 2 * SCREEN_SCALE) + int(SCREEN_SCALE / 2)
        left_foot_z_offset = math.ceil((left_heel.z + left_toe.z) / 2 * SCREEN_SCALE)

        com = get_COM(pose_landmarks, BSP, SCREEN_SCALE) # correct

        face_x_offset, _, face_z_offset = com
        face_x_offset = int(face_x_offset)
        face_y_offset = -int(face_z_offset) + 50


        l_com_dist = get_foot_to_com_distance(left_foot_x_offset, left_foot_z_offset, com)
        r_com_dist = get_foot_to_com_distance(right_foot_x_offset, right_foot_z_offset, com)

        # left heel lifted
        if right_toe.y - left_toe.y > LIFTED_THRESHOLD:
            right_footprint = set_footprint_colour(right_footprint, "red")

        # heel higher, number lower
        # right heel lifted
        elif left_toe.y - right_toe.y > LIFTED_THRESHOLD:
            left_footprint = set_footprint_colour(left_footprint, "red")
        
        #com calculation have issue due to z axis
        # com to right
        elif l_com_dist * COM_DISTANCE_THRESHOLD > r_com_dist:
            right_footprint = set_footprint_colour(right_footprint, "red")

        # # com to left
        elif r_com_dist * COM_DISTANCE_THRESHOLD > l_com_dist:            
            left_footprint = set_footprint_colour(left_footprint, "red")
        
        # offset on screen
        right_y1, right_y2, right_x1, right_x2 = calculate_positions(right_foot_x_offset, right_foot_y_offset, right_footprint.shape)
        left_y1, left_y2, left_x1, left_x2 = calculate_positions(left_foot_x_offset, left_foot_y_offset, left_footprint.shape)
        face_y1, face_y2, face_x1, face_x2 = calculate_positions(face_x_offset, face_y_offset, face.shape)

        # Check if the right_footprint image has an alpha channel
        if right_footprint.shape[2] == 4:
            right_alpha_mask, right_alpha_inv = extract_alpha_masks(right_footprint)
            left_alpha_mask, left_alpha_inv = extract_alpha_masks(left_footprint)
            face_alpha_mask, face_alpha_inv = extract_alpha_masks(face)

            for c in range(3):
                right_crop = img[right_y1:right_y2, right_x1:right_x2, c]
                left_crop = img[left_y1:left_y2, left_x1:left_x2, c]
                face_crop = img[face_y1:face_y2, face_x1:face_x2, c]
                
                dimension_mismatch = any([
                    check_dimensions(right_crop, PIC_SCALE),
                    check_dimensions(left_crop, PIC_SCALE),
                    check_dimensions(face_crop, PIC_SCALE)
                ])
                
                out_of_range = right_y2 > SCREEN_SCALE or left_y2 > SCREEN_SCALE
                
                if dimension_mismatch or out_of_range:
                    cv2.putText(img, "Please stand inside the camera", (0, int(SCREEN_SCALE / 2)), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (0, 0, 255), 4, cv2.LINE_AA)
                    return

                # Right foot overlay
                overlay_image(right_alpha_mask, right_alpha_inv, img, right_footprint, right_y1, right_y2, right_x1, right_x2, c)
                # Left foot overlay
                overlay_image(left_alpha_mask, left_alpha_inv, img, left_footprint, left_y1, left_y2, left_x1, left_x2, c)
                # Face overlay
                overlay_image(face_alpha_mask, face_alpha_inv, img, face, face_y1, face_y2, face_x1, face_x2, c)


    __TRANSFORMERS: Dict[str, Callable[[VideoFrame], VideoFrame]] = {
        "skeleton": _skeleton_transformer,
        "segmentation": _segmentation_transformer,
        "com": _center_of_mass_transformer
    }

    supported_transforms = list(__TRANSFORMERS.keys())

    COM_result = None
    landmarker = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='pose_landmarker_lite.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=__set_COM_result,
    output_segmentation_masks=True,
))
    timestamp = 0

class UnsupportedTransform(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
