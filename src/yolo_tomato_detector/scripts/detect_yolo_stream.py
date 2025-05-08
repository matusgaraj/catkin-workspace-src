#!/usr/bin/env python3
"""
detect_yolo_stream.py: ROS node that
 - subscribes to /camera/color/image_raw
 - runs YOLOv5 detection every 10s
 - draws only “tomato” boxes *and centers* on each frame and republishes to /yolo_detected_stream
 - for each detected tomato computes:
     u,v = bbox center pixel
     r_px = half bbox width in px
     D_mm = raw depth (mm) at (u,v)
   and publishes the list [u,v,r_px,D_mm,…] as Float32MultiArray on /yolo_raw_detections
"""

import rospy, threading, torch, cv2, numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Float32MultiArray

bridge = CvBridge()
latest_frame = None
detection_results = None
lock = threading.Lock()

image_pub = None
raw_pub   = None
model     = None

def draw_boxes(frame, results):
    """
    Draw bounding boxes and centers for each detected tomato.
    """
    boxes = results.xyxy[0].cpu().numpy()
    for x1, y1, x2, y2, conf, cls in boxes:
        label = results.names[int(cls)]
        if label != "tomato":
            continue

        # draw bbox
        x1_i, y1_i, x2_i, y2_i = map(int, (x1, y1, x2, y2))
        cv2.rectangle(frame, (x1_i, y1_i), (x2_i, y2_i), (0,255,0), 2)
        cv2.putText(frame, f"{label}:{conf:.2f}",
                    (x1_i, y1_i - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # compute and draw center
        cx = int(round((x1 + x2) * 0.5))
        cy = int(round((y1 + y2) * 0.5))
        cv2.circle(frame, (cx, cy), 5, (0,255,0), -1)  # red filled circle

    return frame

def image_callback(img_msg):
    global latest_frame, detection_results
    try:
        frame = bridge.imgmsg_to_cv2(img_msg, "bgr8")
        latest_frame = frame.copy()
        out = frame.copy()
        with lock:
            if detection_results is not None:
                out = draw_boxes(out, detection_results)
        image_pub.publish(bridge.cv2_to_imgmsg(out, "bgr8"))
    except Exception as e:
        rospy.logerr(f"[YOLO] image_cb: {e}")

def detection_thread():
    global latest_frame, detection_results
    rate = rospy.Rate(1)  # detection rate
    while not rospy.is_shutdown():
        rate.sleep()
        if latest_frame is None:
            continue

        try:
            # run YOLO
            results = model(latest_frame)
            with lock:
                detection_results = results

            boxes = results.xyxy[0].cpu().numpy()
            raw = []
            # grab one depth frame
            depth_msg = rospy.wait_for_message(
                "/camera/aligned_depth_to_color/image_raw", Image, timeout=5.0)
            depth = bridge.imgmsg_to_cv2(depth_msg, "passthrough")  # uint16 mm

            for x1, y1, x2, y2, conf, cls in boxes:
                label = results.names[int(cls)].lower()
                if label != "tomato":
                    continue
                # bbox center
                u = int(round((x1 + x2) * 0.5))
                v = int(round((y1 + y2) * 0.5))
                # check bounds
                if not (0 <= v < depth.shape[0] and 0 <= u < depth.shape[1]):
                    continue
                D_mm = int(depth[v, u])
                raw.extend([u, v, (x2 - x1) * 0.5, D_mm])

            msg = Float32MultiArray(data=raw)
            raw_pub.publish(msg)

        except Exception as e:
            rospy.logerr(f"[YOLO] detection_thread: {e}")

def main():
    global image_pub, raw_pub, model
    rospy.init_node('yolo_detector', anonymous=False)

    # load model
    model = torch.hub.load(
        'ultralytics/yolov5', 'custom',
        path='/home/matus/catkin_ws/src/yolo_tomato_detector/yolov5/runs/train/exp2/weights/best.pt',
        force_reload=False
    )
    model.eval()

    image_pub = rospy.Publisher("/yolo_detected_stream", Image, queue_size=1)
    raw_pub   = rospy.Publisher("/yolo_raw_detections", Float32MultiArray, queue_size=1)

    rospy.Subscriber("/camera/color/image_raw", Image, image_callback, queue_size=1)
    threading.Thread(target=detection_thread, daemon=True).start()

    rospy.loginfo("[YOLO] node started, waiting for images…")
    rospy.spin()

if __name__=="__main__":
    main()
