#!/usr/bin/env python3
"""
tomato_detector_node.py

A ROS node that loads an image from disk, resizes it to 1200x600 while preserving aspect ratio, 
performs color segmentation to detect red regions (likely tomatoes), finds contours, 
applies Hough Circle Transform, and then refines circle detection using a distance transform-based method.
It outputs several images for inspection.
"""

import rospy            # ROS Python API
import cv2              # OpenCV for image processing
import numpy as np      # NumPy for numerical operations
from skimage.feature import peak_local_max  # For local maximum detection

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image  # For ROS image messages 

def resize_image(image, target_width=1200, target_height=600):
    # Resize an image to fit within 1200x600 while keeping aspect ratio 
    h, w = image.shape[:2]
    scale = min(target_width / w, target_height / h)  # Use min to ensure image fits within target without distortion
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized_image, scale

def process_image(image):
    # Resize image and get scaling factor
    image, scale = resize_image(image)

    # Convert the BGR image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the color ranges for red
    # First red range (low end of hue)
    lower_red1 = np.array([0, 100, 50])
    upper_red1 = np.array([15, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

    # Second red range (high end of hue)
    lower_red2 = np.array([165, 50, 35])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # Combine the two masks using bitwise OR
    mask = mask1 | mask2
    mask_output=mask
    # Clean Up the Mask 
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (60,60))
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    kernel4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30,30))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel4)
    
    mask_output_morf=mask
    # Hough circle transformation pre-processing
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    masked_gray = cv2.bitwise_and(gray_image, gray_image, mask=mask)  
    masked_gray_blurred = cv2.medianBlur(masked_gray, 5)

    # --- Distance Transform & Refinement Section --- #
    # Compute the distance transform from the cleaned mask
    dist_transform = cv2.distanceTransform(masked_gray_blurred, cv2.DIST_L2, 5)
    
    # Suppress pixels above a relative threshold
    global_max = np.max(dist_transform)
    threshold_value = 0.8 * global_max  # Adjust factor as needed (e.g., 0.9 means suppress top 10%)
    dominant_mask = (dist_transform >= threshold_value)
    dist_transform_low = dist_transform.copy()
    dist_transform_low[dominant_mask] = 0

    # Create a depth map (normalized for visualization)
    depth_map = np.uint8(dist_transform_low)
    depth_map = cv2.bitwise_and(gray_image, gray_image, mask=depth_map)
    dist_transform = cv2.distanceTransform(depth_map, cv2.DIST_L2, 5)
    depth_map = np.uint8(dist_transform)
    kernel_erode_depthmap = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (100,100))
    depth_map = cv2.erode(depth_map, kernel_erode_depthmap)
    depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)

    coordinates = peak_local_max(depth_map, min_distance=int(25*scale), threshold_rel=0.4)
    
    # Create a new image to draw circles based on the local maxima.
    dt_output = image.copy()
    for (row, col) in coordinates:
        radius = int(dist_transform[row, col])
        cv2.circle(dt_output, (col, row), radius, (255, 0, 255), 2)
        cv2.circle(dt_output, (col, row), 2, (255, 0, 255), 3)

    # 7) Watershed Section
    # A) Create markers for sure foreground and background
    # Erode or threshold dist_transform to get sure foreground
    sure_fg = np.uint8((dist_transform > 0.5* dist_transform.max()) * 255)  # Adjust factor as needed

    # Sure background can be created by dilating 'mask' or using hull_mask
    kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    sure_bg = cv2.dilate(mask, kernel_bg, iterations=2)

    # Convert sure_bg to binary
    sure_bg = cv2.threshold(sure_bg, 127, 255, cv2.THRESH_BINARY)[1]

    # Unknown region = sure_bg - sure_fg
    unknown = cv2.subtract(sure_bg, sure_fg)

    # B) Connected Components => markers
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1  # Ensure background is not 0

    # Mark unknown region as 0
    markers[unknown == 255] = 0

    # Convert the original image to BGR if needed (already BGR though)
    watershed_input = image.copy()

    # Apply watershed
    markers = cv2.watershed(watershed_input, markers)

    # We'll draw boundaries where markers == -1
    watershed_output = image.copy()
    watershed_output[markers == -1] = [255,  0, 255]  # boundary

    # For visualization or as input to Hough circles, use the refined mask
    mask = np.uint8(markers > 1) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel4)
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    masked_gray = cv2.bitwise_and(gray_image, gray_image, mask=mask)  
    masked_gray_blurred = cv2.medianBlur(masked_gray, 5)
    edges = cv2.Canny(mask, threshold1=50, threshold2=150)

    # Hough circle transformation
    circles = cv2.HoughCircles(
        masked_gray_blurred,             # input image (grayscale)
        cv2.HOUGH_GRADIENT,       # detection method
        dp=1,                     # inverse ratio of accumulator resolution
        minDist=int(200 * scale), # minimum distance between circle centers (Pixels)
        param1=260,               # upper threshold for Canny edge detector
        param2=25,                # threshold for center detection
        minRadius=int(50 * scale),# min circle radius (Pixels)
        maxRadius=1000000         # max circle radius (Pixels)
    )

    # Draw Hough circles if found
    hough_output = image.copy()
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            # Draw the outer circle in green
            cv2.circle(hough_output, center, radius, (0, 255, 0), 2)
            # Draw the center of the circle in green
            cv2.circle(hough_output, center, 2, (0, 255, 0), 3)
     
    # Find Contours 
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter Contours by Area (Optional) 
    min_contour_area = 5000 
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

    # Find Convex Hulls for each contour 
    hull_contours = [cv2.convexHull(cnt) for cnt in filtered_contours]

    # Draw Contours
    output = image.copy()
    hull_output = image.copy()

    # Draw original filtered contours (green)
    cv2.drawContours(output, filtered_contours, -1, (0, 255, 0), 2)

    # Draw convex hull contours (blue)
    cv2.drawContours(hull_output, hull_contours, -1, (255, 0, 0), 2)


    return output, mask, filtered_contours, hull_output, hough_output, masked_gray_blurred, depth_map, dt_output, watershed_output, mask_output, mask_output_morf, edges

def main():
    # Initialize the ROS node
    rospy.init_node('tomato_detector_node', anonymous=True)

    # Set the image path directly in the code
    image_path = "/mnt/c/Users/matga/Desktop/tomato2.jpg"
    rospy.loginfo("Loading image from: %s", image_path)

    # Read the image using OpenCV.
    image = cv2.imread(image_path)
    if image is None:
        rospy.logerr("Could not open or find the image: %s", image_path)
        return

    # Process the image for color segmentation and contour detection.
    (output, mask, filtered_contours, hull_output,
     hough_output, masked_gray_blurred, depth_map, dt_output, watershed_output, mask_output, mask_output_morf, edges) = process_image(image)
     
    rospy.loginfo("Found %d contours", len(filtered_contours))

    # Save images for inspection
    cv2.imwrite("/mnt/c/Users/matga/Desktop/processed_images/Mask.jpg", mask)
    cv2.imwrite("/mnt/c/Users/matga/Desktop/processed_images/Detected_Contours.jpg", output)
    cv2.imwrite("/mnt/c/Users/matga/Desktop/processed_images/Detected_Convex_Hulls.jpg", hull_output)
    cv2.imwrite("/mnt/c/Users/matga/Desktop/processed_images/Grayscale_Masked_Hough.jpg", masked_gray_blurred)
    cv2.imwrite("/mnt/c/Users/matga/Desktop/processed_images/Detected_Hough_Circles.jpg", hough_output)
    cv2.imwrite("/mnt/c/Users/matga/Desktop/processed_images/Depth_Map.jpg", depth_map)
    cv2.imwrite("/mnt/c/Users/matga/Desktop/processed_images/Distance_Transform.jpg", dt_output)
    cv2.imwrite("/mnt/c/Users/matga/Desktop/processed_images/Watershed_Result.jpg", watershed_output)
    cv2.imwrite("/mnt/c/Users/matga/Desktop/processed_images/mask_povodna.jpg", mask_output)
    cv2.imwrite("/mnt/c/Users/matga/Desktop/processed_images/mask_povodna_morf.jpg", mask_output_morf)
    cv2.imwrite("/mnt/c/Users/matga/Desktop/processed_images/edges.jpg", edges)

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
