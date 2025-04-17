#!/usr/bin/env python3
"""
detect_segmentation_stream.py

Subscribes to /camera/color/image_raw at full rate, republishes it on
/seg_detected_stream with your exact static‐image segmentation+DT+watershed+Hough
overlays, and in a background thread (1 Hz) runs the same pipeline to update the
detections and log each circle’s depth from the continuously updated
/camera/depth/image_rect_raw.
"""

import rospy
import threading
import time
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

bridge = CvBridge()
latest_frame = None
latest_depth = None
detected_centers = []  # list of (x,y,radius) in original frame coords
lock = threading.Lock()
depth_lock = threading.Lock()
image_pub = None

def resize_image(image, target_width=1200, target_height=600):
    """Identical to your static script."""
    h, w = image.shape[:2]
    scale = min(target_width / w, target_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA), scale

def process_image(image):
    """
    Mirrors your static tomato_detector_node.py exactly (minus file I/O).
    Returns (resized_output, original_centers).
    """
    # === 1) Resize & HSV segmentation ===
    image_resized, scale = resize_image(image)
    hsv = cv2.cvtColor(image_resized, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 100, 50])
    upper_red1 = np.array([15, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

    lower_red2 = np.array([165, 50, 35])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    mask = mask1 | mask2

    # === 2) Morphological cleanup ===
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (60,60))
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    k4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30,30))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k4)

    # === 3) Masked gray & blur ===
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    mg = cv2.bitwise_and(gray, gray, mask=mask)
    mb = cv2.medianBlur(mg, 5)

    # === 4) Distance Transform & refinement for watershed ===
    dt1 = cv2.distanceTransform(mb, cv2.DIST_L2, 5)
    gm = dt1.max()
    dt1[dt1 >= 0.8 * gm] = 0
    dm = np.uint8(dt1)
    dm = cv2.bitwise_and(gray, gray, mask=dm)
    dist_transform = cv2.distanceTransform(dm, cv2.DIST_L2, 5)

    # === 5) Watershed Section ===
    sure_fg = np.uint8((dist_transform > 0.5 * dist_transform.max()) * 255)
    kb = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    sure_bg = cv2.dilate(mask, kb, iterations=2)
    sure_bg = cv2.threshold(sure_bg, 127, 255, cv2.THRESH_BINARY)[1]
    unknown = cv2.subtract(sure_bg, sure_fg)

    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(image_resized.copy(), markers)

    mask = np.uint8(markers > 1) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2)

    # === 6) Final Hough Circle Transform ===
    mg2   = cv2.bitwise_and(gray, gray, mask=mask)
    edges = cv2.Canny(mg2, 50, 150)
    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=int(200 * scale),
        param1=180,
        param2=20,
        minRadius=int(1 * scale),
        maxRadius=1000000
    )

    # Collect resized centers
    resized_centers = []
    if circles is not None:
        for x_r, y_r, r_r in np.uint16(np.around(circles[0])):
            resized_centers.append((int(x_r), int(y_r), int(r_r)))

    # Shift‐correct back to original frame coords
    original_centers = []
    for x_r, y_r, r_r in resized_centers:
        x_o = int(x_r / scale)
        y_o = int(y_r / scale)
        r_o = int(r_r / scale)
        original_centers.append((x_o, y_o, r_o))

    return image_resized, original_centers

def color_callback(msg):
    """30 Hz subscriber: grab & publish with latest detection overlay."""
    global latest_frame
    try:
        frame = bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError as e:
        rospy.logerr(f"[SegStream] cv_bridge error: {e}")
        return

    with lock:
        latest_frame = frame.copy()
        dets = list(detected_centers)

    vis = frame.copy()
    for x, y, r in dets:
        cv2.circle(vis, (x, y), r, (0, 255, 0), 2)
        cv2.circle(vis, (x, y), 2, (0, 255, 0), 3)

    try:
        image_pub.publish(bridge.cv2_to_imgmsg(vis, "bgr8"))
    except CvBridgeError as e:
        rospy.logerr(f"[SegStream] publish error: {e}")

def depth_callback(msg):
    """Continuously update the latest depth frame."""
    global latest_depth
    try:
        depth = bridge.imgmsg_to_cv2(msg, "passthrough")
    except CvBridgeError:
        return
    with depth_lock:
        latest_depth = depth

def detection_thread():
    """1 Hz detection thread: runs process_image() and logs depth at the same rate."""
    rospy.loginfo("[DetThread] starting up")
    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        rate.sleep()

        with lock:
            frame = None if latest_frame is None else latest_frame.copy()
        if frame is None:
            rospy.logwarn("[DetThread] waiting for first frame")
            continue

        t0 = time.time()
        _, centers = process_image(frame)
        dt_ms = (time.time() - t0) * 1000

        if centers:
            rospy.loginfo(f"[DetThread] {len(centers)} circles in {dt_ms:.1f} ms")
        else:
            rospy.logwarn(f"[DetThread] no circles ({dt_ms:.1f} ms)")

        with lock:
            detected_centers[:] = centers

        # log depth
        with depth_lock:
            dimg = None if latest_depth is None else latest_depth.copy()

        if dimg is None:
            rospy.logwarn("[DetThread] no depth frame yet")
            continue

        h, w = dimg.shape[:2]
        k = 5
        yy, xx = np.ogrid[-k:k+1, -k:k+1]
        circle_mask = (yy*yy + xx*xx) <= (k*k)

        for (x, y, r) in centers:
            if y < k or y >= h-k or x < k or x >= w-k:
                rospy.logwarn(f"[DetThread] center ({x},{y}) out of bounds")
                continue

            window = dimg[y-k:y+k+1, x-k:x+k+1]
            vals = window[circle_mask]
            valid = vals[vals > 0]
            if valid.size == 0:
                rospy.logwarn(f"[DetThread] no valid depth around ({x},{y})")
                continue

            z = np.median(valid) * 0.001
            rospy.loginfo(f"[DetThread] Tomato @({x},{y}) ≈{z:.2f} m")

def main():
    global bridge, image_pub
    rospy.init_node("seg_tomato_detector_stream", anonymous=True)
    bridge = CvBridge()
    image_pub = rospy.Publisher("/seg_detected_stream", Image, queue_size=1)

    # subscribe to color and aligned‐depth streams
    rospy.Subscriber("/camera/color/image_raw", Image, color_callback, queue_size=1)
    rospy.Subscriber("/camera/depth/image_rect_raw", Image, depth_callback, queue_size=1)

    threading.Thread(target=detection_thread, daemon=True).start()
    rospy.spin()

if __name__ == "__main__":
    main()
