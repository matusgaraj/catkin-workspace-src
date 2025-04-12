#!/usr/bin/env python3

import os
import subprocess
import rospy

def train_yolo():
    rospy.init_node('yolo_tomato_trainer', anonymous=True)
    
    # Define the paths:
    # Assume you have YOLOv5 code in a submodule folder 'yolov5' inside your package
    package_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    yolov5_dir = os.path.join(package_dir, "yolov5")
    
    # Data config file inside your package's config folder
    data_config = os.path.join(package_dir, "config", "data.yaml")  # Updated for your dataset
    
    # Choose a model config file from YOLOv5 (for example, yolov5m.yaml or yolov5s.yaml)
    model_config = os.path.join(yolov5_dir, "models", "yolov5m.yaml")  # You can change 'yolov5m.yaml' to 'yolov5s.yaml'
    
    # Pre-trained weights (assume they're located in the yolov5 folder)
    weights = os.path.join(yolov5_dir, "yolov5m.pt")  # You can change 'yolov5m.pt' to another pre-trained model
    
    # Build the training command with additional parameters for batch size, epochs, and early stopping:
    train_cmd = [
        "python3", os.path.join(yolov5_dir, "train.py"),
        "--img", "640",  # Image size
        "--batch", "8",  # Batch size set to 16 (adjustable)
        "--epochs", "50",  # Number of epochs (adjustable)
        "--data", data_config,  # Path to the dataset YAML file
        "--cfg", model_config,  # Path to the model config file
        "--weights", weights,  # Path to pre-trained weights
        "--device", "0",  # Use GPU device 0 (change if using multiple GPUs)
        "--patience", "5" , # Patience for early stopping (training will stop if validation loss doesn't improve for 10 epochs)
        "--resume"  # Resume training from last checkpoint
    ]   
    
    rospy.loginfo("Starting YOLOv5 training...")
    
    # Run the training command:
    ret = subprocess.call(train_cmd)
    
    if ret != 0:
        rospy.logerr("Training failed with error code: {}".format(ret))
    else:
        rospy.loginfo("Training finished successfully.")

if __name__ == "__main__":
    try:
        train_yolo()
    except rospy.ROSInterruptException:
        pass
