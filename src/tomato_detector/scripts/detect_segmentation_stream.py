#!/usr/bin/env python3
"""
detect_segmentation_stream.py

Subscribes to /camera/color/image_raw and /camera/aligned_depth_to_color/image_raw,
runs your segmentation+DT+watershed+Hough pipeline, overlays circles+centers on
/seg_detected_stream at full rate, and at 1 Hz publishes [u,v,r_mm,D_mm] on
/yolo_raw_detections for the common coordinate_transform pipeline to consume.
"""

import rospy, threading, time
import cv2, numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32MultiArray

# Globals
bridge = CvBridge()
latest_frame = None
latest_depth = None
detected_centers = []  # list of (x_o, y_o, r_o) in original image coords
lock = threading.Lock()
depth_lock = threading.Lock()
image_pub = None
raw_pub   = None

def resize_image(image, target_width=1200, target_height=600):
    """Resize while preserving aspect, return (resized, scale)."""
    h, w = image.shape[:2]
    scale = min(target_width / w, target_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale

def process_image(image):
    """
    Run your full segmentation pipeline on a resized copy.
    Returns a list of (x_o, y_o, r_o) in the ORIGINAL image coordinates.
    """
    # 1) Resize & HSV thresholds
    image_resized, scale = resize_image(image)
    hsv = cv2.cvtColor(image_resized, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 100, 50])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([165,  50,  35])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 | mask2

    # 2) Morphological cleanup
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (60,60))
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    k4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30,30))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k4)

    # 3) Prepare for watershed
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    mg = cv2.bitwise_and(gray, gray, mask=mask)
    mb = cv2.medianBlur(mg, 5)

    dt1 = cv2.distanceTransform(mb, cv2.DIST_L2, 5)
    gm = dt1.max()
    dt1[dt1 >= 0.8 * gm] = 0
    dm = np.uint8(dt1)
    dm = cv2.bitwise_and(gray, gray, mask=dm)
    dist_transform = cv2.distanceTransform(dm, cv2.DIST_L2, 5)

    sure_fg = np.uint8((dist_transform > 0.5 * dist_transform.max()) * 255)
    kb = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    sure_bg = cv2.dilate(mask, kb, iterations=2)
    sure_bg = cv2.threshold(sure_bg, 127, 255, cv2.THRESH_BINARY)[1]
    unknown = cv2.subtract(sure_bg, sure_fg)

    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(image_resized.copy(), markers)

    mask2 = np.uint8(markers > 1) * 255
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN,  k1)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, k2)

    # 4) Hough circles on cleaned region
    mg2   = cv2.bitwise_and(gray, gray, mask=mask2)
    edges = cv2.Canny(mg2, 50, 150)
    circles = cv2.HoughCircles(
        edges, cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=int(200 * scale),
        param1=180,
        param2=20,
        minRadius=int(1 * scale),
        maxRadius=1000000
    )

    # 5) Collect and un-scale centers
    original_centers = []
    if circles is not None:
        for x_r, y_r, r_r in np.uint16(np.around(circles[0])):
            # back up to original image coords
            x_o = int(x_r / scale)
            y_o = int(y_r / scale)
            r_o = int(r_r / scale)
            original_centers.append((x_o, y_o, r_o))

    return original_centers


def color_callback(msg):
    """30 Hz: overlay circles+centers on the live color stream."""
    global latest_frame, detected_centers
    try:
        frame = bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError as e:
        rospy.logerr(f"[Seg] image_cb: {e}")
        return

    with lock:
        latest_frame = frame.copy()
        dets = list(detected_centers)

    vis = frame.copy()
    for x, y, r in dets:
        cv2.circle(vis, (x, y), r, (0, 255, 0), 2)  # circle edge
        cv2.circle(vis, (x, y),   2, (0,   255, 0), -1)  # center dot

    try:
        image_pub.publish(bridge.cv2_to_imgmsg(vis, "bgr8"))
    except CvBridgeError as e:
        rospy.logerr(f"[Seg] publish error: {e}")


def depth_callback(msg):
    """Update the latest aligned-depth frame."""
    global latest_depth
    try:
        d = bridge.imgmsg_to_cv2(msg, "passthrough")
    except CvBridgeError:
        return
    with depth_lock:
        latest_depth = d.copy()


def detection_thread():
    """1 Hz: run segmentation, update centers, and publish raw [u,v,r_px,D_mm]."""
    global latest_frame, latest_depth, detected_centers, raw_pub

    rate = rospy.Rate(1)  
    k = 5
    yy, xx = np.ogrid[-k:k+1, -k:k+1]
    circle_mask = (yy*yy + xx*xx) <= (k*k)

    rospy.loginfo("[DetThread] starting")
    while not rospy.is_shutdown():
        rate.sleep()

        # grab a frame
        with lock:
            if latest_frame is None:
                rospy.logwarn("[DetThread] waiting for color frame")
                continue
            frame = latest_frame.copy()

        # run segmentation → scaled to original coords
        centers = process_image(frame)
        with lock:
            detected_centers[:] = centers

        # grab depth
        with depth_lock:
            if latest_depth is None:
                rospy.logwarn("[DetThread] waiting for depth frame")
                continue
            depth_img = latest_depth.copy()

        raw = []
        h, w = depth_img.shape[:2]
        for (x_o, y_o, r_o) in centers:
            # bounds check
            if x_o < k or x_o >= w-k or y_o < k or y_o >= h-k:
                rospy.logwarn(f"[DetThread] center out of bounds: {x_o},{y_o}")
                continue

            window = depth_img[y_o-k:y_o+k+1, x_o-k:x_o+k+1]
            vals = window[circle_mask]
            valid = vals[vals > 0]
            if valid.size == 0:
                rospy.logwarn(f"[DetThread] no valid depth around {x_o},{y_o}")
                continue

            z_mm = float(np.median(valid))
            # publish in same format as YOLO: [u, v, r_px, depth_mm]
            raw.extend([float(x_o), float(y_o), float(r_o), z_mm])

        if raw:
            raw_pub.publish(Float32MultiArray(data=raw))


def main():
    global image_pub, raw_pub
    rospy.init_node("seg_tomato_detector_stream", anonymous=True)

    # 1) overlay publisher
    image_pub = rospy.Publisher("/seg_detected_stream", Image, queue_size=1)
    # 2) raw data for tf pipeline
    raw_pub   = rospy.Publisher("/yolo_raw_detections",
                                Float32MultiArray, queue_size=1)

    # 3) subscribers
    rospy.Subscriber("/camera/color/image_raw", Image,
                     color_callback, queue_size=1)
    rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image,
                     depth_callback, queue_size=1)

    # 4) start 1 Hz detection thread
    threading.Thread(target=detection_thread, daemon=True).start()

    rospy.loginfo("[SegStream] node started")
    rospy.spin()


if __name__ == "__main__":
    main()
