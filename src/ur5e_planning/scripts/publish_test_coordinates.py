#!/usr/bin/env python3
"""
publish_test_marker.py: Publikuje 3D marker s nastaviteľným polomerom na vizualizáciu paradajky v RViz.
"""
import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

def main():
    rospy.init_node("publish_test_coordinates", anonymous=True)
    pub = rospy.Publisher("/tomato_position", Marker, queue_size=10)
    rate = rospy.Rate(0.05)  # Hz

    radius = 0.03  # in meters (30 mm)

    while not rospy.is_shutdown():
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "test_tomato"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        # Position
        marker.pose.position.x = 0.5
        marker.pose.position.y = 0.5
        marker.pose.position.z = 0.5

        # Orientation (no rotation)
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        # Scale (diameter in each dimension)
        marker.scale.x = radius * 2
        marker.scale.y = radius * 2
        marker.scale.z = radius * 2

        # Color (red tomato)
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0  # alpha (opacity)

        marker.lifetime = rospy.Duration()  # 0 = forever
        pub.publish(marker)

        rospy.loginfo("Publikujem marker s polomerom %.2f m na pozícii (%.2f, %.2f, %.2f)",
                      radius, marker.pose.position.x, marker.pose.position.y, marker.pose.position.z)

        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
