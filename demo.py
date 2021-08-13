#!/usr/bin/env python3

import argparse
from MovenetRenderer import MovenetRenderer


parser = argparse.ArgumentParser()
parser.add_argument("-e", "--edge", action="store_true",
                    help="Use Edge mode (the cropping algorithm runs on device)")
parser.add_argument("-m", "--model", type=str, default='thunder',
                    help="Model to use : 'thunder' or 'lightning' or path of a blob file (default=%(default)s)")
parser.add_argument('-i', '--input', type=str, default='rgb',
                    help="'rgb' or 'rgb_laconic' or path to video/image file to use as input (default=%(default)s)")
parser.add_argument('-c', '--crop', action="store_true", 
                    help="Center cropping frames to a square shape (smaller size of original frame)") 
parser.add_argument('-nsc', '--no_smart_crop', action="store_true", 
                    help="Disable smart cropping from previous frame detection")   
parser.add_argument("-s", "--score_threshold", default=0.2, type=float,
                    help="Confidence score to determine whether a keypoint prediction is reliable (default=%(default)f)") 
parser.add_argument('-f', '--internal_fps', type=int,                                                                                     
                    help="Fps of internal color camera. Too high value lower NN fps (default: depends on the model")    
parser.add_argument('--internal_frame_height', type=int, default=640,                                                                                    
                    help="Internal color camera frame height in pixels (default=%(default)i)")          
parser.add_argument("-o","--output",
                    help="Path to output video file")

    
args = parser.parse_args()

if args.edge:
    from MovenetDepthaiEdge import MovenetDepthai
else:
    from MovenetDepthai import MovenetDepthai

pose = MovenetDepthai(input_src=args.input, 
            model=args.model,    
            score_thresh=args.score_threshold,  
            crop=args.crop,    
            smart_crop=not args.no_smart_crop,     
            internal_fps=args.internal_fps,
            internal_frame_height=args.internal_frame_height
            )

renderer = MovenetRenderer(
                pose, 
                output=args.output)

while True:
    # Run movenet on next frame
    frame, body = pose.next_frame()
    if frame is None: break
    # Draw 2d skeleton
    frame = renderer.draw(frame, body)
    key = renderer.waitKey(delay=1)
    if key == 27 or key == ord('q'):
        break
renderer.exit()
pose.exit()