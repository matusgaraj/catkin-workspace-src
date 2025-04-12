#!/usr/bin/env python3
"""
publish_test_coordinates.py: ROS uzol, ktorý publikuje testovacie 3D súradnice paradajky.
Publikuje správu typu PointStamped na tému /tomato_position.
Tento uzol slúži na ladenie a overenie transformácií v RViz.
"""
import rospy
from geometry_msgs.msg import PointStamped

def main():
    rospy.init_node("publish_test_coordinates", anonymous=True)
    pub = rospy.Publisher("/tomato_position", PointStamped, queue_size=10)
    rate = rospy.Rate(0.05)  # 1 Hz pre testovanie
    while not rospy.is_shutdown():
        point_msg = PointStamped()
        point_msg.header.stamp = rospy.Time.now()
        point_msg.header.frame_id = "base_link"  # alebo iný rámec, s ktorým pracuješ
        # Testovacie súradnice – uprav podľa potreby
        point_msg.point.x = 0.5
        point_msg.point.y = 0.5
        point_msg.point.z = 0.5
        rospy.loginfo("Publikujem testovacie súradnice: (%.3f, %.3f, %.3f)", 
                      point_msg.point.x, point_msg.point.y, point_msg.point.z)
        pub.publish(point_msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
