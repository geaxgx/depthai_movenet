import cv2
import numpy as np

# LINES_*_BODY are used when drawing the skeleton onto the source image. 
# Each variable is a list of continuous lines.
# Each line is a list of keypoints as defined at https://github.com/tensorflow/tfjs-models/tree/master/pose-detection#keypoint-diagram

LINES_BODY = [[4,2],[2,0],[0,1],[1,3],
                [10,8],[8,6],[6,5],[5,7],[7,9],
                [6,12],[12,11],[11,5],
                [12,14],[14,16],[11,13],[13,15]]

class MovenetRenderer:
    def __init__(self,
                pose,
                output=None):
        self.pose = pose

        # Rendering flags
        self.show_fps = True
        self.show_crop = False

        if output is None:
            self.output = None
        else:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            self.output = cv2.VideoWriter(output,fourcc,pose.video_fps,(pose.img_w, pose.img_h))

    def draw(self, frame, body):
        self.frame = frame
        lines = [np.array([body.keypoints[point] for point in line]) for line in LINES_BODY if body.scores[line[0]] > self.pose.score_thresh and body.scores[line[1]] > self.pose.score_thresh]
        cv2.polylines(frame, lines, False, (255, 180, 90), 2, cv2.LINE_AA)
        
        for i,x_y in enumerate(body.keypoints):
            if body.scores[i] > self.pose.score_thresh:
                if i % 2 == 1:
                    color = (0,255,0) 
                elif i == 0:
                    color = (0,255,255)
                else:
                    color = (0,0,255)
                cv2.circle(frame, (x_y[0], x_y[1]), 4, color, -11)

        if self.show_crop:
            cv2.rectangle(frame, (body.crop_region.xmin, body.crop_region.ymin), (body.crop_region.xmax, body.crop_region.ymax), (0,255,255), 2)

        return frame

    def exit(self):
        if self.output:
            self.output.release()

    def waitKey(self, delay=1):
        if self.show_fps:
                self.pose.fps.draw(self.frame, orig=(50,50), size=1, color=(240,180,100))
        cv2.imshow("Movenet", self.frame)
        if self.output:
            self.output.write(self.frame)
        key = cv2.waitKey(delay) 
        if key == 32:
            # Pause on space bar
            cv2.waitKey(0)
        elif key == ord('f'):
            self.show_fps = not self.show_fps
        elif key == ord('c'):
            self.show_crop = not self.show_crop
        return key