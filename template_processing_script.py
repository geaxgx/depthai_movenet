"""
This file is the template of the scripting node source code in edge mode
Substitution is made in MovenetDepthaiEdge.py
"""
import marshal

def torso_visible(scores):
    """Checks whether there are enough torso keypoints.

    This function checks whether the model is confident at predicting one of the
    shoulders/hips which is required to determine a good crop region.
    """
    return ((scores[11] > ${_score_thresh} or
            scores[12] > ${_score_thresh}) and
            (scores[5] > ${_score_thresh} or
            scores[6] > ${_score_thresh}))

def determine_torso_and_body_range(x, y, scores, center_x, center_y):
    """Calculates the maximum distance from each keypoints to the center location.

    The function returns the maximum distances from the two sets of keypoints:
    full 17 keypoints and 4 torso keypoints. The returned information will be
    used to determine the crop size. See determine_crop_region for more detail.
    """
    torso_joints = [5, 6, 11, 12]
    max_torso_yrange = 0.0
    max_torso_xrange = 0.0
    for i in torso_joints:
        dist_y = abs(center_y - y[i])
        dist_x = abs(center_x - x[i])
        if dist_y > max_torso_yrange:
            max_torso_yrange = dist_y
        if dist_x > max_torso_xrange:
            max_torso_xrange = dist_x

    max_body_yrange = 0.0
    max_body_xrange = 0.0
    for i in range(17):
        if scores[i] < ${_score_thresh}:
            continue
        dist_y = abs(center_y - y[i])
        dist_x = abs(center_x - x[i])
        if dist_y > max_body_yrange:
            max_body_yrange = dist_y
        if dist_x > max_body_xrange:
            max_body_xrange = dist_x

    return [max_torso_yrange, max_torso_xrange, max_body_yrange, max_body_xrange]

def determine_crop_region(scores, x, y):
    """Determines the region to crop the image for the model to run inference on.

    The algorithm uses the detected joints from the previous frame to estimate
    the square region that encloses the full body of the target person and
    centers at the midpoint of two hip joints. The crop size is determined by
    the distances between each joints and the center point.
    When the model is not confident with the four torso joint predictions, the
    function returns a default crop which is the full image padded to square.
    """
    if torso_visible(scores):
        center_x = (x[11] + x[12]) // 2
        center_y = (y[11] + y[12]) // 2
        max_torso_yrange, max_torso_xrange, max_body_yrange, max_body_xrange = determine_torso_and_body_range(x, y, scores, center_x, center_y)
        crop_length_half = max(max_torso_xrange * 1.9, max_torso_yrange * 1.9, max_body_yrange * 1.2, max_body_xrange * 1.2)
        crop_length_half = int(round(min(crop_length_half, max(center_x, ${_img_w} - center_x, center_y, ${_img_h} - center_y))))
        crop_corner = [center_x - crop_length_half, center_y - crop_length_half]

        if crop_length_half > max(${_img_w}, ${_img_h}) / 2:
            return ${_init_crop_region}
        else:
            crop_length = crop_length_half * 2
            return {'xmin': crop_corner[0], 'ymin': crop_corner[1], 'xmax': crop_corner[0]+crop_length, 'ymax': crop_corner[1]+crop_length, 'size': crop_length}
    else:
        return ${_init_crop_region}

def pd_postprocess(inference, crop_region):
    size = crop_region['size']
    xmin = crop_region['xmin']
    ymin = crop_region['ymin']
    xnorm = []
    ynorm = []
    scores = []
    x = []
    y = []
    for i in range(17):
        xn = inference[3*i+1]
        yn = inference[3*i]
        xnorm.append(xn)
        ynorm.append(yn)
        scores.append(inference[3*i+2])
        x.append(int(xmin + xn * size)) 
        y.append(int(ymin + yn * size)) 
          
    next_crop_region = determine_crop_region(scores, x, y) if ${_smart_crop} else init_crop_region
    return x, y, xnorm, ynorm, scores, next_crop_region

node.warn("Processing node started")
# Defines the default crop region (pads the full image from both sides to make it a square image) 
# Used when the algorithm cannot reliably determine the crop region from the previous frame.
init_crop_region = ${_init_crop_region}
crop_region = init_crop_region
result_buffer = Buffer(759)
while True:
    # Send cropping information to manip node on device
    cfg = ImageManipConfig()
    points = [
        [crop_region['xmin'], crop_region['ymin']],
        [crop_region['xmax']-1, crop_region['ymin']],
        [crop_region['xmax']-1, crop_region['ymax']-1],
        [crop_region['xmin'], crop_region['ymax']-1]]
    point2fList = []
    for p in points:
        pt = Point2f()
        pt.x, pt.y = p[0], p[1]
        point2fList.append(pt)
    cfg.setWarpTransformFourPoints(point2fList, False)
    cfg.setResize(${_pd_input_length}, ${_pd_input_length})
    cfg.setFrameType(ImgFrame.Type.RGB888p)
    node.io['to_manip_cfg'].send(cfg)

    # Receive movenet inference
    inference = node.io['from_pd_nn'].get().getLayerFp16("Identity")
    # Process it
    x, y, xnorm, ynorm, scores, next_crop_region = pd_postprocess(inference, crop_region)
    # Send result to the host
    result = {"x":x, "y":y, "xnorm":xnorm, "ynorm":ynorm, "scores":scores, "next_crop_region":next_crop_region}
    result_serial = marshal.dumps(result)
    # Uncomment the following line to know the correct size of result_buffer
    # node.warn("result_buffer size: "+str(len(result_serial)))

    result_buffer.getData()[:] = result_serial
    node.io['to_host'].send(result_buffer)

    crop_region = next_crop_region

