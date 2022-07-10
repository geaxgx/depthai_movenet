#!/usr/bin/env python3

import cv2
from math import atan2, degrees
import sys
sys.path.append("../..")
from MovenetDepthai import MovenetDepthai, KEYPOINT_DICT
from MovenetRenderer import MovenetRenderer
import argparse
import numpy as np

def estimate_focus_zone_size(body, scale, score_thresh):
    """
    This function estimate the zine of the zone.
    We calculate the length of segments from a predefined list. A segment length
    is the distance between the 2 endpoints weighted by a coefficient. The weight have been chosen
    so that the weighted length length of all segments are roughly equal. 
    We take the maximal length to estimate the size of the focus zone. 
    If no segment are visible, we consider the body is very close 
    to the camera, and therefore there is no need to focus. Return 0
    To not have at least one shoulder and one hip visible means the body is also very close
    and the estimated size needs to be adjusted (bigger)
    """
    segments = [
        ("left_shoulder", "left_elbow", 2.3),
        ("left_elbow", "left_wrist", 2.3),
        ("left_shoulder", "left_hip", 1),
        ("left_shoulder", "right_shoulder", 1.5),
        ("right_shoulder", "right_elbow", 2.3),
        ("right_elbow", "right_wrist", 2.3),
        ("right_shoulder", "right_hip", 1),
    ]
    lengths = []
    for s in segments:
        if body.scores[KEYPOINT_DICT[s[0]]] > score_thresh and body.scores[KEYPOINT_DICT[s[1]]] > score_thresh:
            l = np.linalg.norm(body.keypoints[KEYPOINT_DICT[s[0]]] - body.keypoints[KEYPOINT_DICT[s[1]]])
            lengths.append(l)
    if lengths:   
        if ( body.scores[KEYPOINT_DICT["left_hip"]] < score_thresh and
            body.scores[KEYPOINT_DICT["right_hip"]] < score_thresh or
            body.scores[KEYPOINT_DICT["left_shoulder"]] < score_thresh and
            body.scores[KEYPOINT_DICT["right_shoulder"]] < score_thresh) :
            coef = 1.5
        else:
            coef = 1.0
        return 2 * int(coef * scale * max(lengths) / 2) # The size is made even
    else:
        return 0

def get_focus_zone(body, frame, hand_label, scale, score_thresh, hands_up_only = False):
    """
    Return a list of zones around one or 2 hands depending on the value of hand_label.
    If hand_label == "left_right", the list contains at most 2 zones, the zone around the left hand, 
    and the zone around the right hand.
    For all othe values of hand_label, the list contains at most one zone.
    If hand_value == "left" (resp "right"), this is the zone around the left (resp right) hand.
    If hand_value == "higher", this is the zone around the hand closest to the top of the image.
    If hand_value == "group", this a larger zone that contains both hands, or only one hand if an hand is not visible.
    A zone is a list [left, top, right, bottom] describing the position in pixels of the zone in the image.
    """

    def zone_from_center_size(x, y, size):
        """
        Calculate the top left corner (x1, y1) and botom right corner (x2, y2) of the zone from its center and size
        """
        half_size = size // 2
        size = half_size * 2
        x1 = x - half_size
        x2 = x + half_size -1
        if x1 < 0:
            x1 = 0
            x2 = size - 1
        if x2 >= w:
            x2 = w - 1
            x1 = w - size
        y1 = y - half_size
        y2 = y + half_size
        if y1 < 0:
            y1 = 0
            y2 = size - 1
        if y2 >= h:
            y2 = h - 1
            y1 = h - size
        return [x1, y1, x2, y2]

    def get_one_hand_zone(hand_label, scale, hands_up_only):
        """
        Determine the zone around the "hand_label" (left of right) hand.
        Return [left, top, right, bottom] of the zone
        or None if the zone could not be determined
        """
        wrist_kp = hand_label + "_wrist"
        wrist_score = body.scores[KEYPOINT_DICT[wrist_kp]]
        if wrist_score < score_thresh: 
            return None
        x, y = body.keypoints[KEYPOINT_DICT[wrist_kp]]
        if hands_up_only:
            # We want to detect only hands where the wrist is above the elbow (when visible)
            elbow_kp = hand_label + "_elbow"
            if body.scores[KEYPOINT_DICT[elbow_kp]] > score_thresh and \
                body.keypoints[KEYPOINT_DICT[elbow_kp]][1] < body.keypoints[KEYPOINT_DICT[wrist_kp]][1]:
                return None
        # Let's evaluate the size of the focus zone
        size = estimate_focus_zone_size(body, scale, score_thresh)
        if size == 0: return [0, 0, frame_size-1, frame_size-1] # The hand is too close. No need to focus
        return zone_from_center_size(x, y, size)


    h,w = frame.shape[:2]
    frame_size = max(h, w)
    zone_list = []
    if hand_label == "group":
        zonel = get_one_hand_zone("left", scale, hands_up_only)
        if zonel:
            zoner = get_one_hand_zone("right", scale, hands_up_only)
            if zoner:
                xl1, yl1, xl2, yl2 = zonel
                xr1, yr1, xr2, yr2 = zoner
                x1 = min(xl1, xr1)
                y1 = min(yl1, yr1)
                x2 = max(xl2, xr2)
                y2 = max(yl2, yr2)
                # Center (x,y)
                x = int((x1+x2)/2)
                y = int((y1+y2)/2)
                size_x = x2-x1
                size_y = y2-y1
                size = 2 * (max(size_x, size_y) // 2)
                zone_list.append([zone_from_center_size(x, y, size), "group"])
            else:
                zone_list.append([zonel, "left"])
        else:
            zoner = get_one_hand_zone("right", scale, hands_up_only)
            if zoner:
                zone_list.append([zoner, "right"])

    elif hand_label == "higher":
        if body.scores[KEYPOINT_DICT["left_wrist"]] > score_thresh:
            if body.scores[KEYPOINT_DICT["right_wrist"]] > score_thresh:
                if body.keypoints[KEYPOINT_DICT["left_wrist"]][1] > body.keypoints[KEYPOINT_DICT["right_wrist"]][1]:
                    hand_label = "right"
                else:
                    hand_label = "left"
            else: 
                 hand_label = "left"
        else:
            if body.scores[KEYPOINT_DICT["right_wrist"]] > score_thresh:
                hand_label = "right"
            else:
                return []
        zone = get_one_hand_zone(hand_label, scale, hands_up_only)
        if zone: zone_list.append([zone, hand_label])
    elif hand_label == "left_right":
        zoner = get_one_hand_zone("right", scale, hands_up_only)
        if zoner: zone_list.append([zoner, "right"])
        zonel = get_one_hand_zone("left", scale, hands_up_only)
        if zonel: zone_list.append([zonel, "left"])
    else: # "left" or "right"
        zone_list.append([get_one_hand_zone(hand_label, scale, hands_up_only), hand_label])
    return zone_list
           
            

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, choices=['lightning', 'thunder'], default='thunder',
                        help="Model to use (default=%(default)s)")
parser.add_argument("-f", "--focus", type=str, choices=['left', 'right', 'group', 'higher', 'left_right'], default='higher',
                        help="Find square zone(s) around hand(s) (default=%(default)s)")
parser.add_argument("-s", "--scale", type=float, default=1.0,
                    help="Zone scaling factor (default=%(default)f)")
parser.add_argument('-u', '--hands_up_only', action="store_true", 
                    help="Take into considerations only the hands where the wrist is above the elbow")                     
parser.add_argument('-c', '--crop', action="store_true", 
                    help="Center cropping frames to a square shape (smaller size of original frame)") 
parser.add_argument('-nsc', '--no_smart_crop', action="store_true", 
                    help="Disable smart cropping from previous frame detection")  
parser.add_argument('-i', '--input', type=str, default='rgb',
                    help="'rgb' or 'rgb_laconic' or path to video/image file to use as input (default: %(default)s)")  
parser.add_argument("-o","--output",
                    help="Path to output video file")
args = parser.parse_args()            

pose = MovenetDepthai(
    input_src=args.input, 
    model=args.model,
    crop=args.crop,
    smart_crop=not args.no_smart_crop
)
score_thresh = pose.score_thresh
renderer = MovenetRenderer(pose, output=args.output)
renderer.show_fps = False

nb = 0
while True:
    # Run blazepose on next frame
    frame, body = pose.next_frame()
    if frame is None: break
    # Get the focus zone around the hand or hands we are interested in
    result = get_focus_zone(body, frame, args.focus, args.scale, score_thresh, args.hands_up_only)
    for zone, hand_label in result:
        if zone:
            if hand_label == "group":
                color = (255,0,0)
            elif hand_label == "right":
                color = (0,0,255)
            else: # left
                color = (0,255,0)
            cv2.rectangle(frame, tuple(zone[:2]), tuple(zone[2:]), color, 3)
    # Draw 2d skeleton
    frame = renderer.draw(frame, body)
    
    key = renderer.waitKey(delay=1)
    if key == 27 or key == ord('q'):
        break
renderer.exit()
pose.exit()

