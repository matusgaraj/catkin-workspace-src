#!/usr/bin/env python3
"""
detect_yolo_stream.py: ROS uzol, ktorý odoberá obrazový stream zo zariadenia Realsense (farebný stream z /camera/color/image_raw),
vykonáva YOLOv5 detekciu raz za 10 sekúnd a prekresľuje bounding boxy na každom frame. 
Po detekcii získava stred bounding boxu, získa hĺbkovú hodnotu z depth obrazu (/camera/image_rect_raw),
a vypíše 3D hĺbkové údaje do terminálu. Ak nie je detekovaná žiadna paradajka, vypíše správu.
"""

import rospy
import torch
import cv2
import time
import threading
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np

# Globálne premenné pre uchovávanie posledného získaného frame a výsledkov detekcie
latest_frame = None         # Najnovší farebný frame (z /camera/color/image_raw)
detection_results = None    # Najnovšie výsledky YOLO detekcie (bounding boxy, confidence, trieda)
lock = threading.Lock()     # Lock na zabezpečenie prístupu k detection_results

bridge = CvBridge()
image_pub = None

def draw_boxes(frame, results):
    """
    Prekreslí bounding boxy na daný frame podľa výsledkov YOLO.
    YOLOv5 výsledky sú v results.xyxy[0]: [x1, y1, x2, y2, confidence, class].
    """
    try:
        boxes = results.xyxy[0].cpu().numpy()
    except Exception as e:
        rospy.logerr("Chyba pri konverzii detekčných výsledkov: " + str(e))
        return frame

    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        label = results.names[int(cls)]
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        text = f"{label}: {conf:.2f}"
        cv2.putText(frame, text, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def image_callback(data):
    global latest_frame, detection_results
    try:
        # Konvertujeme ROS Image na OpenCV obraz (BGR formát)
        frame = bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
        latest_frame = frame.copy()
        
        # Prekreslíme bounding boxy, ak máme detekčné výsledky
        output_frame = frame.copy()
        with lock:
            if detection_results is not None:
                output_frame = draw_boxes(output_frame, detection_results)
        
        # Publikujeme upravený frame (s prekreslenými bounding boxmi)
        image_pub.publish(bridge.cv2_to_imgmsg(output_frame, encoding="bgr8"))
    except Exception as e:
        rospy.logerr(f"Chyba v image callback: {e}")

def detection_thread():
    global latest_frame, detection_results
    while not rospy.is_shutdown():
        # Čakáme 10 sekúnd medzi detekciami
        rospy.sleep(1.0)
        if latest_frame is not None:
            try:
                # Spustíme YOLO detekciu na najnovšom frame
                results = model(latest_frame)
                with lock:
                    detection_results = results

                # Získame bounding boxy z výsledkov
                boxes = results.xyxy[0].cpu().numpy()
                if boxes.size == 0:
                    rospy.loginfo("Nedetekovaná žiadna paradajka.")
                else:
                    # Pokusíme sa nájsť najbližšiu paradajku (len tých s labelom "tomato")
                    closest_depth = None
                    closest_center = None
                    # Získame depth obrázok zo správnej témy:
                    depth_msg = rospy.wait_for_message("/camera/depth/image_rect_raw", Image, timeout=5.0)
                    depth_image = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
                    
                    for box in boxes:
                        x1, y1, x2, y2, conf, cls = box
                        label = results.names[int(cls)].lower()
                        if label != "tomato":
                            continue
                        # Vypočítame stred bounding boxu
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        # Získame hĺbkovú hodnotu pre stred (depth_image má hodnoty v milimetroch)
                        depth_value = depth_image[center_y, center_x]
                        # Prepočítame na metre 
                        depth_m = depth_value * 0.001
                        # Overíme, že hodnota je kladná (platná)
                        if depth_m <= 0.001:
                            continue
                        if closest_depth is None or depth_m < closest_depth:
                            closest_depth = depth_m
                            closest_center = (center_x, center_y)
                    
                    if closest_depth is not None:
                        rospy.loginfo("Najbližšia paradajka: stred (%d, %d), vzdialenosť: %.3f m",
                                      closest_center[0], closest_center[1], closest_depth)
                    else:
                        rospy.loginfo("Nedetekovaná žiadna paradajka (len 'unripe' alebo neplatné hĺbky).")
            except Exception as e:
                rospy.logerr("Chyba v detekčnom vlákne: %s", e)


def main():
    global model, image_pub
    rospy.init_node('yolo_tomato_detector_stream', anonymous=True)
    
    # Načítanie YOLOv5 modelu s tvojimi natrénovanými váhami
    model = torch.hub.load('ultralytics/yolov5', 'custom', 
                            path='/home/matus/catkin_ws/src/yolo_tomato_detector/yolov5/runs/train/exp2/weights/best.pt',
                            force_reload=False)
    model.eval()  # Nastavíme model do eval režimu

    # Publisher pre upravený obrazový stream
    image_pub = rospy.Publisher('/yolo_detected_stream', Image, queue_size=10)
    
    # Odoberáme farebný obrazový stream z Realsense (predpokladáme tému /camera/color/image_raw)
    rospy.Subscriber('/camera/color/image_raw', Image, image_callback, queue_size=1)
    
    rospy.loginfo("YOLO detekčný uzol pre Realsense stream spustený, čakám na obraz...")
    
    # Spustenie detekčného vlákna, ktoré beží raz za 10 sekúnd
    threading.Thread(target=detection_thread, daemon=True).start()
    
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
