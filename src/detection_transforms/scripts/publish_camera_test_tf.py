#!/usr/bin/env python3
import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from visualization_msgs.msg import Marker
from math import radians
from tf.transformations import quaternion_from_euler

def publish_static_transforms(bcaster):
    # 1) base_link → wrist_3_link (identity)
    t0 = TransformStamped()
    t0.header.frame_id    = "base_link"
    t0.child_frame_id     = "wrist_3_link"
    t0.header.stamp       = rospy.Time.now()
    t0.transform.translation.x = 0
    t0.transform.translation.y = 0
    t0.transform.translation.z = 0
    t0.transform.rotation.x = 0
    t0.transform.rotation.y = 0
    t0.transform.rotation.z = 0
    t0.transform.rotation.w = 1
    bcaster.sendTransform(t0)

    # 2) wrist_3_link → camera_link
    #    CAD offsets: left, above, front
    tx, ty, tz = -0.017500, -0.076287, 0.032679

    # Correct 8° downward tilt: rotate about camera X axis by –8°
    roll  = radians(-8.0)
    pitch = 0.0
    yaw   = 0.0
    qx, qy, qz, qw = quaternion_from_euler(roll, pitch, yaw)

    t1 = TransformStamped()
    t1.header.frame_id    = "wrist_3_link"
    t1.child_frame_id     = "camera_link"
    t1.header.stamp       = rospy.Time.now()
    t1.transform.translation.x = tx
    t1.transform.translation.y = ty
    t1.transform.translation.z = tz
    t1.transform.rotation.x    = qx
    t1.transform.rotation.y    = qy
    t1.transform.rotation.z    = qz
    t1.transform.rotation.w    = qw
    bcaster.sendTransform(t1)

def publish_test_marker(pub):
    m = Marker()
    m.header.frame_id = "camera_link"
    m.header.stamp    = rospy.Time.now()
    m.ns, m.id        = "camera_test", 0
    m.type            = Marker.SPHERE
    m.action          = Marker.ADD

    # After the static transform (with roll = –8°), this
    # point will land at (0,0,0.30) in wrist_3_link’s Z.
    m.pose.position.x = 0.01750
    m.pose.position.y = 0.03834
    m.pose.position.z = 0.27534

    m.scale.x = m.scale.y = m.scale.z = 0.05
    m.color.r = 1.0; m.color.a = 1.0
    pub.publish(m)


if __name__=="__main__":
    rospy.init_node("test_camera_tf", anonymous=True)
    bcaster = tf2_ros.StaticTransformBroadcaster()
    rospy.sleep(0.5)
    publish_static_transforms(bcaster)

    marker_pub = rospy.Publisher("/test_pt", Marker,
                                 queue_size=1, latch=True)
    rospy.sleep(0.5)
    publish_test_marker(marker_pub)

    rospy.loginfo("Published wrist_3_link→camera_link with –8° roll and test sphere.")
    rospy.spin()
