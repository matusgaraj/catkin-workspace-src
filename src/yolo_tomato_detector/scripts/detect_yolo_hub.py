#!/usr/bin/env python3
"""
detect_hub.py: A ROS node to run YOLOv5 detection using PyTorch Hub.
This script loads the model via PyTorch Hub and runs inference on an image.
"""

import torch
import cv2
import rospy
from sensor_msgs.msg import Image  # Optional, if you integrate with ROS image messages
from cv_bridge import CvBridge
from std_msgs.msg import String

def detect_image(image_path):
    # Load YOLOv5 model using PyTorch Hub
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/matus/catkin_ws/src/yolo_tomato_detector/yolov5/runs/train/exp2/weights/best.pt', force_reload=False)
    
    # Run inference
    results = model(image_path)
    
    # Print or process results
    print(results)
    # Display results (if needed)
    results.show()  # This opens a window with detections, but might not work in headless WSL environments
    # Alternatively, save results
    results.save(save_dir='/mnt/c/Users/matga/Desktop/yolo_result_hub.jpg')
    
    return results

def main():
    rospy.init_node('yolo_tomato_detector_hub', anonymous=True)
    pub = rospy.Publisher('detection_results', String, queue_size=10)
    bridge = CvBridge()
    
    # Replace with your test image path
    image_path = "/mnt/c/Users/matga/Desktop/tomatoes/tomato.jpg"
    results = detect_image(image_path)
    
    # Publish a simple message with results summary
    pub.publish("Detection complete using PyTorch Hub. Check console and output folder.")
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
