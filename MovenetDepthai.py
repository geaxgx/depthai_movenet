import numpy as np
import cv2
from collections import namedtuple
from pathlib import Path
from FPS import FPS
import depthai as dai
import time

from math import gcd

SCRIPT_DIR = Path(__file__).resolve().parent
MOVENET_LIGHTNING_MODEL = SCRIPT_DIR / "models/movenet_singlepose_lightning_U8_transpose.blob"
MOVENET_THUNDER_MODEL = SCRIPT_DIR / "models/movenet_singlepose_thunder_U8_transpose.blob"

# Dictionary that maps from joint names to keypoint indices.
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

class Body:
    def __init__(self, scores=None, keypoints_norm=None, keypoints=None, score_thresh=None, crop_region=None, next_crop_region=None):
        """
        Attributes:
        scores : scores of the keypoints
        keypoints_norm : keypoints normalized ([0,1]) coordinates (x,y) in the squared cropped region
        keypoints : keypoints coordinates (x,y) in pixels in the source image
        score_thresh : score threshold used
        crop_region : cropped region on which the current body was inferred
        next_crop_region : cropping region calculated from the current body keypoints and that will be used on next frame
        """
        self.scores = scores 
        self.keypoints_norm = keypoints_norm 
        self.keypoints = keypoints
        self.score_thresh = score_thresh
        self.crop_region = crop_region
        self.next_crop_region = next_crop_region

    def print(self):
        attrs = vars(self)
        print('\n'.join("%s: %s" % item for item in attrs.items()))

CropRegion = namedtuple('CropRegion',['xmin', 'ymin', 'xmax',  'ymax', 'size']) # All values are in pixel. The region is a square of size 'size' pixels

def find_isp_scale_params(size, is_height=True):
    """
    Find closest valid size close to 'size' and and the corresponding parameters to setIspScale()
    This function is useful to work around a bug in depthai where ImageManip is scrambling images that have an invalid size
    is_height : boolean that indicates if the value is the height or the width of the image
    Returns: valid size, (numerator, denominator)
    """
    # We want size >= 288
    if size < 288:
        size = 288
    
    # We are looking for the list on integers that are divisible by 16 and
    # that can be written like n/d where n <= 16 and d <= 63
    if is_height:
        reference = 1080 
        other = 1920
    else:
        reference = 1920 
        other = 1080
    size_candidates = {}
    for s in range(288,reference,16):
        f = gcd(reference, s)
        n = s//f
        d = reference//f
        if n <= 16 and d <= 63 and int(round(other * n / d) % 2 == 0):
            size_candidates[s] = (n, d)
            
    # What is the candidate size closer to 'size' ?
    min_dist = -1
    for s in size_candidates:
        dist = abs(size - s)
        if min_dist == -1:
            min_dist = dist
            candidate = s
        else:
            if dist > min_dist: break
            candidate = s
            min_dist = dist
    return candidate, size_candidates[candidate]

    

class MovenetDepthai:
    """
    Movenet body pose detector
    Arguments:
    - input_src: frame source, 
                    - "rgb" or None: OAK* internal color camera,
                    - "rgb_laconic": same as "rgb" but without sending the frames to the host,
                    - a file path of an image or a video,
                    - an integer (eg 0) for a webcam id,
    - model: Movenet blob file,
                    - "thunder": the default thunder blob file (see variable MOVENET_THUNDER_MODEL),
                    - "lightning": the default lightning blob file (see variable MOVENET_LIGHTNING_MODEL),
                    - a path of a blob file. It is important that the filename contains 
                    the string "thunder" or "lightning" to identify the tyoe of the model.
    - score_thresh : confidence score to determine whether a keypoint prediction is reliable (a float between 0 and 1).
    - crop : boolean which indicates if systematic square cropping to the smaller side of 
                    the image is done or not,
    - smart_crop : boolen which indicates if cropping from previous frame detection is done or not,
    - internal_fps : when using the internal color camera as input source, set its FPS to this value (calling setFps()).
    - internal_frame_height : when using the internal color camera, set the frame height (calling setIspScale()).
                                The width is calculated accordingly to height and depends on value of 'crop'
    - stats : True or False, when True, display the global FPS when exiting.            
    """
    def __init__(self, input_src="rgb",
                model=None, 
                score_thresh=0.2,
                crop=False,
                smart_crop = True,
                internal_fps=None,
                internal_frame_height=640,
                stats=True):
        

        self.model = model 
        
        
        if model == "lightning":
            self.model = str(MOVENET_LIGHTNING_MODEL)
            self.pd_input_length = 192
        elif model == "thunder":
            self.model = str(MOVENET_THUNDER_MODEL)
            self.pd_input_length = 256
        else:
            self.model = model
            if "lightning" in str(model):
                self.pd_input_length = 192
            else: # Thunder
                self.pd_input_length = 256
        print(f"Using blob file : {self.model}")

        print(f"MoveNet imput size : {self.pd_input_length}x{self.pd_input_length}x3")
        self.score_thresh = score_thresh   
        
        self.crop = crop
        self.smart_crop = smart_crop
        self.internal_fps = internal_fps
        self.stats = stats
        
        if input_src is None or input_src == "rgb" or input_src == "rgb_laconic":
            self.input_type = "rgb" # OAK* internal color camera
            self.laconic = "laconic" in input_src # Camera frames are not sent to the host
            if internal_fps is None:
                if "thunder" in str(model):
                    self.internal_fps = 12
                else:
                    self.internal_fps = 26
            else:
                self.internal_fps = internal_fps
            print(f"Internal camera FPS set to: {self.internal_fps}")

            self.video_fps = self.internal_fps # Used when saving the output in a video file. Should be close to the real fps
            
            if self.crop:
                self.frame_size, self.scale_nd = find_isp_scale_params(internal_frame_height)
                self.img_h = self.img_w = self.frame_size
            else:
                width, self.scale_nd = find_isp_scale_params(internal_frame_height * 1920 / 1080, is_height=False)
                self.img_h = int(round(1080 * self.scale_nd[0] / self.scale_nd[1]))
                self.img_w = int(round(1920 * self.scale_nd[0] / self.scale_nd[1]))
                self.frame_size = self.img_w
            print(f"Internal camera image size: {self.img_w} x {self.img_h}")

        elif input_src.endswith('.jpg') or input_src.endswith('.png'):
            self.input_type= "image"
            self.img = cv2.imread(input_src)
            self.video_fps = 25
            self.img_h, self.img_w = self.img.shape[:2]
        else:
            self.input_type = "video"
            if input_src.isdigit():
                input_type = "webcam"
                input_src = int(input_src)
            self.cap = cv2.VideoCapture(input_src)
            self.video_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            self.img_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.img_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print("Video FPS:", self.video_fps)

        # Defines the default crop region (pads the full image from both sides to make it a square image) 
        # Used when the algorithm cannot reliably determine the crop region from the previous frame.
        box_size = max(self.img_w, self.img_h)
        x_min = (self.img_w - box_size) // 2
        y_min = (self.img_h - box_size) // 2
        self.init_crop_region = CropRegion(x_min, y_min, x_min+box_size, y_min+box_size, box_size)
        self.crop_region = self.init_crop_region
        

        self.device = dai.Device(self.create_pipeline())
        # self.device.startPipeline()
        print("Pipeline started")

        # Define data queues 
        if self.input_type == "rgb":
            if not self.laconic:
                self.q_video = self.device.getOutputQueue(name="cam_out", maxSize=1, blocking=False)
            self.q_manip_cfg = self.device.getInputQueue(name="manip_cfg")
        else:
            self.q_pd_in = self.device.getInputQueue(name="pd_in")
        self.q_pd_out = self.device.getOutputQueue(name="pd_out", maxSize=4, blocking=False)
            
        self.fps = FPS()

        self.nb_frames = 0
        self.nb_pd_inferences = 0

    def create_pipeline(self):
        print("Creating pipeline...")
        # Start defining a pipeline
        pipeline = dai.Pipeline()
        pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2021_3)

        if self.input_type == "rgb":
            # ColorCamera
            print("Creating Color Camera...")
            # cam = pipeline.create(dai.node.ColorCamera) 
            cam = pipeline.createColorCamera()
            cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
            cam.setInterleaved(False)
            cam.setIspScale(self.scale_nd[0], self.scale_nd[1])
            cam.setFps(self.internal_fps)
            cam.setBoardSocket(dai.CameraBoardSocket.RGB)
            cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
            if self.crop:
                cam.setVideoSize(self.frame_size, self.frame_size)
                cam.setPreviewSize(self.frame_size, self.frame_size)
            else: 
                cam.setVideoSize(self.img_w, self.img_h)
                cam.setPreviewSize(self.img_w, self.img_h)

            if not self.laconic:
                cam_out = pipeline.createXLinkOut()
                # cam_out = pipeline.create(dai.node.XLinkOut)
                cam_out.setStreamName("cam_out")
                cam.video.link(cam_out.input)

            # ImageManip for cropping
            manip = pipeline.createImageManip()
            manip.setMaxOutputFrameSize(self.pd_input_length*self.pd_input_length*3)
            manip.setWaitForConfigInput(True)
            manip.inputImage.setQueueSize(1)
            manip.inputImage.setBlocking(False)            

            manip_cfg = pipeline.createXLinkIn()
            manip_cfg.setStreamName("manip_cfg")

            cam.preview.link(manip.inputImage)
            manip_cfg.out.link(manip.inputConfig)

        # Define pose detection model
        print("Creating Pose Detection Neural Network...")
        pd_nn = pipeline.createNeuralNetwork()
        # pd_nn = pipeline.create(dai.node.NeuralNetwork)
        pd_nn.setBlobPath(str(Path(self.model).resolve().absolute()))
        # pd_nn.input.setQueueSize(1)
        # Increase threads for detection
        # pd_nn.setNumInferenceThreads(2)
        if self.input_type == "rgb":
            manip.out.link(pd_nn.input)
        else:
            pd_in = pipeline.createXLinkIn()
            pd_in.setStreamName("pd_in")
            pd_in.out.link(pd_nn.input)

        # Define link to send pd result to host 
        pd_out = pipeline.createXLinkOut()
        # pd_out = pipeline.create(dai.node.XLinkOut)
        pd_out.setStreamName("pd_out")

        pd_nn.out.link(pd_out.input)

        print("Pipeline created.")

        return pipeline        
    
    def crop_and_resize(self, frame, crop_region):
        """Crops and resize the image to prepare for the model input."""
        cropped = frame[max(0,crop_region.ymin):min(self.img_h,crop_region.ymax), max(0,crop_region.xmin):min(self.img_w,crop_region.xmax)]
        
        if crop_region.xmin < 0 or crop_region.xmax >= self.img_w or crop_region.ymin < 0 or crop_region.ymax >= self.img_h:
            # Padding is necessary        
            cropped = cv2.copyMakeBorder(cropped, 
                                        max(0,-crop_region.ymin), 
                                        max(0, crop_region.ymax-self.img_h),
                                        max(0,-crop_region.xmin), 
                                        max(0, crop_region.xmax-self.img_w),
                                        cv2.BORDER_CONSTANT)

        cropped = cv2.resize(cropped, (self.pd_input_length, self.pd_input_length), interpolation=cv2.INTER_AREA)
        return cropped

    def torso_visible(self, scores):
        """Checks whether there are enough torso keypoints.

        This function checks whether the model is confident at predicting one of the
        shoulders/hips which is required to determine a good crop region.
        """
        return ((scores[KEYPOINT_DICT['left_hip']] > self.score_thresh or
                scores[KEYPOINT_DICT['right_hip']] > self.score_thresh) and
                (scores[KEYPOINT_DICT['left_shoulder']] > self.score_thresh or
                scores[KEYPOINT_DICT['right_shoulder']] > self.score_thresh))

    def determine_torso_and_body_range(self, keypoints, scores, center_x, center_y):
        """Calculates the maximum distance from each keypoints to the center location.

        The function returns the maximum distances from the two sets of keypoints:
        full 17 keypoints and 4 torso keypoints. The returned information will be
        used to determine the crop size. See determine_crop_region for more detail.
        """
        torso_joints = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
        max_torso_yrange = 0.0
        max_torso_xrange = 0.0
        for joint in torso_joints:
            dist_y = abs(center_y - keypoints[KEYPOINT_DICT[joint]][1])
            dist_x = abs(center_x - keypoints[KEYPOINT_DICT[joint]][0])
            if dist_y > max_torso_yrange:
                max_torso_yrange = dist_y
            if dist_x > max_torso_xrange:
                max_torso_xrange = dist_x

        max_body_yrange = 0.0
        max_body_xrange = 0.0
        for i in range(len(KEYPOINT_DICT)):
            if scores[i] < self.score_thresh:
                continue
            dist_y = abs(center_y - keypoints[i][1])
            dist_x = abs(center_x - keypoints[i][0])
            if dist_y > max_body_yrange:
                max_body_yrange = dist_y
            if dist_x > max_body_xrange:
                max_body_xrange = dist_x

        return [max_torso_yrange, max_torso_xrange, max_body_yrange, max_body_xrange]

    def determine_crop_region(self, body):
        """Determines the region to crop the image for the model to run inference on.

        The algorithm uses the detected joints from the previous frame to estimate
        the square region that encloses the full body of the target person and
        centers at the midpoint of two hip joints. The crop size is determined by
        the distances between each joints and the center point.
        When the model is not confident with the four torso joint predictions, the
        function returns a default crop which is the full image padded to square.
        """
        if self.torso_visible(body.scores):
            center_x = (body.keypoints[KEYPOINT_DICT['left_hip']][0] + body.keypoints[KEYPOINT_DICT['right_hip']][0]) // 2
            center_y = (body.keypoints[KEYPOINT_DICT['left_hip']][1] + body.keypoints[KEYPOINT_DICT['right_hip']][1]) // 2
            max_torso_yrange, max_torso_xrange, max_body_yrange, max_body_xrange = self.determine_torso_and_body_range(body.keypoints, body.scores, center_x, center_y)
            crop_length_half = np.amax([max_torso_xrange * 1.9, max_torso_yrange * 1.9, max_body_yrange * 1.2, max_body_xrange * 1.2])
            tmp = np.array([center_x, self.img_w - center_x, center_y, self.img_h - center_y])
            crop_length_half = int(round(np.amin([crop_length_half, np.amax(tmp)])))
            crop_corner = [center_x - crop_length_half, center_y - crop_length_half]

            if crop_length_half > max(self.img_w, self.img_h) / 2:
                return self.init_crop_region
            else:
                crop_length = crop_length_half * 2
                return CropRegion(crop_corner[0], crop_corner[1], crop_corner[0]+crop_length, crop_corner[1]+crop_length,crop_length)
        else:
            return self.init_crop_region

    def pd_postprocess(self, inference):
        kps = np.array(inference.getLayerFp16('Identity')).reshape(-1,3) # 17x3
        body = Body(scores=kps[:,2], keypoints_norm=kps[:,[1,0]], score_thresh=self.score_thresh)
        body.keypoints = (np.array([self.crop_region.xmin, self.crop_region.ymin]) + body.keypoints_norm * self.crop_region.size).astype(np.int)
        body.crop_region = self.crop_region
        if self.smart_crop:
            body.next_crop_region = self.determine_crop_region(body)
        return body

    def next_frame(self):

        self.fps.update()
        if self.input_type == "rgb":
            # Send cropping information to manip node on device
            cfg = dai.ImageManipConfig()
            points = [
                [self.crop_region.xmin, self.crop_region.ymin],
                [self.crop_region.xmax-1, self.crop_region.ymin],
                [self.crop_region.xmax-1, self.crop_region.ymax-1],
                [self.crop_region.xmin, self.crop_region.ymax-1]]
            point2fList = []
            for p in points:
                pt = dai.Point2f()
                pt.x, pt.y = p[0], p[1]
                point2fList.append(pt)
            cfg.setWarpTransformFourPoints(point2fList, False)
            cfg.setResize(self.pd_input_length, self.pd_input_length)
            cfg.setFrameType(dai.ImgFrame.Type.RGB888p)
            self.q_manip_cfg.send(cfg)

            # Get the device camera frame if wanted
            if self.laconic:
                frame = np.zeros((self.frame_size, self.frame_size, 3), dtype=np.uint8)
            else:
                in_video = self.q_video.get()
                frame = in_video.getCvFrame()
        else:
            if self.input_type == "image":
                frame = self.img.copy()
            else:
                ok, frame = self.cap.read()
                if not ok:
                    return None, None
                
            # Cropping of the video frame
            cropped = self.crop_and_resize(frame, self.crop_region)
            cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB).transpose(2,0,1)
            
            frame_nn = dai.ImgFrame()
            frame_nn.setTimestamp(time.monotonic())
            frame_nn.setWidth(self.pd_input_length)
            frame_nn.setHeight(self.pd_input_length)
            frame_nn.setData(cropped)
            self.q_pd_in.send(frame_nn)

        # Get result from device
        inference = self.q_pd_out.get()
        body = self.pd_postprocess(inference)
        if self.smart_crop:
            self.crop_region = body.next_crop_region

        # Statistics
        if self.stats:
            self.nb_frames += 1
            self.nb_pd_inferences += 1


        return frame, body


    def exit(self):
        # Print some stats
        if self.stats:
            print(f"FPS : {self.fps.global_duration():.1f} f/s (# frames = {self.fps.nb_frames()})")
           