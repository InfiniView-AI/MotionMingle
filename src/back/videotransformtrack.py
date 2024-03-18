from typing import Dict, Callable

import cv2
import mediapipe as mp
from aiortc import MediaStreamTrack
from av import VideoFrame
import numpy as np
import COM

from mediapipe.framework.formats import landmark_pb2
import math

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
            condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.5
            ouput_frame = np.where(condition, frame, frame_rgb * 0.3).astype(np.uint8)

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
    def __draw_landmarks_on_image(rgb_image, detection_result):
        pose_landmarks_list = detection_result.pose_landmarks
        annotated_image = np.copy(rgb_image)

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

            cv2.rectangle(annotated_image,(0,0),(200,200),(255,255,255),-1)

            VideoTransformTrack.draw_COM(annotated_image, pose_landmarks, 1)


        return annotated_image

    @staticmethod
    def _set_COM_result(result: PoseLandmarkerResult, *args):
        VideoTransformTrack.COM_result = result

    
    @staticmethod
    def draw_COM(img, pose_landmarks, t):
        screen_scale = 200
        pic_scale = 40
        def getCOM(pose_landmarks):
            left_waist = pose_landmarks[23]
            right_waist = pose_landmarks[24]
            mean_x = (left_waist.x + right_waist.x) / 2 * screen_scale
            mean_y = (left_waist.y + right_waist.y) / 2 * screen_scale
            mean_z = (left_waist.z + right_waist.z) / 2 * screen_scale

            return (mean_x, mean_y, mean_z)
            
        def rotateImage(image, angle):
            row = image.shape[0]
            col = image.shape[1]
            center=tuple(np.array([row,col])/2)
            rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
            new_image = cv2.warpAffine(image, rot_mat, (col,row))
            return new_image

        def get_foot_angle(foot_heel, foot_toe):
            return math.degrees(math.atan2(foot_heel.z - foot_toe.z, foot_heel.x - foot_toe.x)) + 90

        
        def get_foot_to_com_distance(x, z, com):
            return (x - com[0]) **2 + (z - com[2]) ** 2
    
        def set_footprint_colour(footprint, colour):
            if colour == "blue":
                footprint[:, :, 0] = 255
                footprint[:, :, 1] = 255
                footprint[:, :, 2] = 0
            elif colour == "orange":
                footprint[:, :, 0] = 0
                footprint[:, :, 1] = 165
                footprint[:, :, 2] = 255
            elif colour == "red":
                footprint[:, :, 1] = 0
                footprint[:, :, 0] = 0
                footprint[:, :, 2] = 255

            return footprint

        def get_foot_to_com_distance(x, z, com):
            return abs((x - com[0]) **2 + (z - com[2]) ** 2)



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
        face = cv2.resize(cv2.imread("face.png", -1), (pic_scale, pic_scale))
        face = rotateImage(face, shoulder_angle + 90)
        
        # resize, change colour and rotate
        right_footprint = cv2.resize(right_footprint, (pic_scale, pic_scale))
        right_footprint = set_footprint_colour(right_footprint, "blue")
        right_footprint = rotateImage(right_footprint, right_foot_angle)

        left_footprint = cv2.resize(left_footprint, (pic_scale, pic_scale))
        left_footprint = rotateImage(left_footprint, left_foot_angle)
        left_footprint = set_footprint_colour(left_footprint, "orange")

        # calculate offset
        right_foot_x_offset = math.ceil((right_heel.x + right_toe.x) / 2 * screen_scale)
        right_foot_y_offset = -math.ceil((right_heel.z + right_toe.z) / 2 * screen_scale) + int(screen_scale / 2)
        right_foot_z_offset = math.ceil((right_heel.z + right_toe.z) / 2 * screen_scale)

        left_foot_x_offset = math.ceil((left_heel.x + left_toe.x) / 2 * screen_scale) # feet width adjustment
        left_foot_y_offset = -math.ceil((left_heel.z + left_toe.z) / 2 * screen_scale) + int(screen_scale / 2)
        left_foot_z_offset = math.ceil((left_heel.z + left_toe.z) / 2 * screen_scale)

        com = getCOM(pose_landmarks) # correct

        face_x_offset, _, face_z_offset = com
        face_x_offset = int(face_x_offset)
        face_y_offset = -int(face_z_offset) + 100

        lifted_threshold = 0.025

        l_com_dist = get_foot_to_com_distance(left_foot_x_offset, left_foot_z_offset, com)
        r_com_dist = get_foot_to_com_distance(right_foot_x_offset, right_foot_z_offset, com)

        # left heel lifted
        if right_heel.y - left_heel.y > lifted_threshold:
            right_footprint = set_footprint_colour(right_footprint, "red")

        # heel higher, number lower
        # right heel lifted
        elif left_heel.y - right_heel.y > lifted_threshold:
            left_footprint = set_footprint_colour(left_footprint, "red")

        
        #com calculation have issue due to z axis
        # com to right
        elif l_com_dist * 0.6 > r_com_dist:
            right_footprint = set_footprint_colour(right_footprint, "red")


        # # com to left
        elif r_com_dist * 0.6 > l_com_dist:            
            left_footprint = set_footprint_colour(left_footprint, "red")
        
        # offset on screen
        right_y1, right_y2 = right_foot_y_offset, right_foot_y_offset + right_footprint.shape[0]
        right_x1, right_x2 = right_foot_x_offset, right_foot_x_offset + right_footprint.shape[1]

        left_y1, left_y2 = left_foot_y_offset , left_foot_y_offset + left_footprint.shape[0]
        left_x1, left_x2 = left_foot_x_offset , left_foot_x_offset + left_footprint.shape[1]

        face_y1, face_y2 = face_y_offset , face_y_offset + face.shape[0]
        face_x1, face_x2 = face_x_offset , face_x_offset + face.shape[1]


        if right_footprint.shape[2] == 4:
            # Extract the alpha mask
            right_alpha_mask = right_footprint[:, :, 3] / 255.0
            right_alpha_inv = 1.0 - right_alpha_mask

            left_alpha_mask = left_footprint[:, :, 3] / 255.0
            left_alpha_inv = 1.0 - left_alpha_mask


        if right_footprint.shape[2] == 4:
            # Extract the alpha mask
            right_alpha_mask = right_footprint[:, :, 3] / 255.0
            right_alpha_inv = 1.0 - right_alpha_mask

            left_alpha_mask = left_footprint[:, :, 3] / 255.0
            left_alpha_inv = 1.0 - left_alpha_mask

            face_alpha_mask = face[:, :, 3] / 255.0
            face_alpha_inv = 1.0 - face_alpha_mask


            # TODO: feet position is back side
            # Split the background and overlay in 3 channels
            for c in range(0, 3):
                # TODO: diff dimension
                right_bool = img[right_y1:right_y2, right_x1:right_x2, c].shape[0] != pic_scale or img[right_y1:right_y2, right_x1:right_x2, c].shape[1] != pic_scale
                left_bool = img[left_y1:left_y2, left_x1:left_x2, c].shape[0] != pic_scale or img[left_y1:left_y2, left_x1:left_x2, c].shape[1] != pic_scale
                face_bool = img[face_y1:face_y2, face_x1:face_x2, c].shape[0] != pic_scale or img[face_y1:face_y2, face_x1:face_x2, c].shape[1] != pic_scale
                right_out_of_range = right_y2 > screen_scale
                left_out_of_range = left_y2 > screen_scale
                if right_bool or left_bool or face_bool or right_out_of_range or left_out_of_range:
                    return

                # print(right_alpha_inv * img[right_y1:right_y2, right_x1:right_x2, c])
                # right foot
                img[right_y1:right_y2, right_x1:right_x2, c] = (right_alpha_mask * right_footprint[:, :, c] +
                                          right_alpha_inv * img[right_y1:right_y2, right_x1:right_x2, c])
                # left foot
                img[left_y1:left_y2, left_x1:left_x2, c] = (left_alpha_mask * left_footprint[:, :, c] +
                                          left_alpha_inv * img[left_y1:left_y2, left_x1:left_x2, c])
                
        
                img[face_y1:face_y2, face_x1:face_x2, c] = (face_alpha_mask * face[:, :, c] + 
                                        face_alpha_inv * img[face_y1: face_y2, face_x1: face_x2, c])


    __TRANSFORMERS: Dict[str, Callable[[VideoFrame], VideoFrame]] = {
        "cartoon": _cartoon_transformer,
        "edges": _edges_transformer,
        "rotate": _rotate_transformer,
        "skeleton": _skeleton_transformer,
        "segmentation": _segmentation_transformer,
        "com": _center_of_mass_transformer
    }

    supported_transforms = list(__TRANSFORMERS.keys())

    COM_result = None
    landmarker = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='pose_landmarker_lite.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=_set_COM_result,
    output_segmentation_masks=True,
))
    timestamp = 0

class UnsupportedTransform(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
