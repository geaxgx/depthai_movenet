from MovenetDepthai import MovenetDepthai, MOVENET_THUNDER_MODEL, MOVENET_LIGHTNING_MODEL
from MovenetRenderer import MovenetRenderer
import argparse




parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, choices=['lightning', 'thunder'], default='thunder',
                        help="Model to use (default=%(default)s")
parser.add_argument('-i', '--input', type=str, default='rgb',
                    help="'rgb' or 'rgb_laconic' or path to video/image file to use as input (default: %(default)s)")
parser.add_argument("-b", "--blob", type=str,
                    help="Path to an .blob file for pose detection model")
# parser.add_argument('-c', '--crop', action="store_true", 
#                     help="Center crop frames to a square shape before feeding pose detection model")   
parser.add_argument("-s", "--score_threshold", default=0.2, type=float,
                        help="Confidence score to determine whether a keypoint prediction is reliable (default=%(default)f)") 
parser.add_argument('--internal_fps', type=int,                                                                                     
                    help="Fps of internal color camera. Too high value lower NN fps (default=%(default)i)")    
parser.add_argument('--internal_frame_size', type=int, default=640,                                                                                    
                    help="Internal color camera frame size (= width = height) in pixels (default=%(default)i)")          
parser.add_argument("-o","--output",
                    help="Path to output video file")

    

args = parser.parse_args()

if args.blob is None:
    if args.model == "thunder":
        args.blob = MOVENET_THUNDER_MODEL
    else:
        args.blob = MOVENET_LIGHTNING_MODEL

pose = MovenetDepthai(input_src=args.input, 
            blob=args.blob,    
            score_thresh=args.score_threshold,           
            internal_fps=args.internal_fps,
            internal_frame_size=args.internal_frame_size
            )

renderer = MovenetRenderer(
                pose, 
                output=args.output)

while True:
    # Run blazepose on next frame
    frame, body = pose.next_frame()
    if frame is None: break
    # Draw 2d skeleton
    frame = renderer.draw(frame, body)
    key = renderer.waitKey(delay=1)
    if key == 27 or key == ord('q'):
        break
renderer.exit()
pose.exit()