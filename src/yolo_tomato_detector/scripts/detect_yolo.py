#!/usr/bin/env python3
"""
detect_yolo.py: A ROS node to run YOLOv5 detection on a given image or video source.
This script calls the YOLOv5 detect.py script using subprocess and logs the results.
"""

import os
import subprocess
import rospy
from std_msgs.msg import String  # You can use a more specific message type if needed

def run_detection():
    rospy.loginfo("Starting YOLOv5 detection...")

    # Define the paths:
    # Assume YOLOv5 code is inside your package in the 'yolov5' folder
    package_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    yolov5_dir = os.path.join(package_dir, "yolov5")

    # Path to the trained weights, adjust as necessary (using the best checkpoint)
    weights_path = os.path.join(yolov5_dir, "runs", "train", "exp2", "weights", "best.pt")

    # Source for detection: here, we use a test image. You can change this to a video or ROS image topic.
    # For now, let's use a local test image path.
    source = "/mnt/c/Users/matga/Desktop/tomatoes/tomato6.jpg"  # Update this path

    # Build the detection command:
    detect_cmd = [
        "python3", os.path.join(yolov5_dir, "detect.py"),
        "--weights", weights_path,
        "--img", "640",
        "--conf", "0.25",
        "--source", source  # can be an image file, a video file, or a camera index
    ]

    rospy.loginfo("Running detection command: " + " ".join(detect_cmd))
    
    try:
        # Run the detection command and capture output
        output = subprocess.check_output(detect_cmd, stderr=subprocess.STDOUT)
        rospy.loginfo("Detection output:\n" + output.decode())
    except subprocess.CalledProcessError as e:
        rospy.logerr("Detection failed with error: " + e.output.decode())

def main():
    rospy.init_node('yolo_tomato_detector', anonymous=True)
    # Optionally, you can set up a publisher to publish detection results
    pub = rospy.Publisher('detection_results', String, queue_size=10)

    run_detection()

    # For this example, simply publish a log message
    pub.publish("Detection complete. Check the output folder for results.")
    rospy.spin()  # Keep the node alive if needed

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
