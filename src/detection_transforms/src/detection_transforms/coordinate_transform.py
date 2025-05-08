#!/usr/bin/env python3
import rospy
import numpy as np
import tf2_ros
import tf2_geometry_msgs               # for do_transform_pose
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import PoseStamped, TransformStamped
from visualization_msgs.msg import Marker
from tf.transformations import quaternion_from_euler
from image_geometry import PinholeCameraModel
import math

class CameraIntrinsics:
    def __init__(self, fx, fy, cx, cy, frame_id):
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.frame_id = frame_id

intr           = None
tf_buffer      = None
marker_pub     = None
static_bcaster = None
cam_model      = None

def camera_info_cb(msg: CameraInfo):
    global intr, tf_buffer, static_bcaster, cam_model
    if intr is None:
        # load aligned‐depth intrinsics
        cam_model = PinholeCameraModel()
        cam_model.fromCameraInfo(msg)
        intr = CameraIntrinsics(
            cam_model.fx(),
            cam_model.fy(),
            cam_model.cx(),
            cam_model.cy(),
            msg.header.frame_id
        )

        # TF listener
        tf_buffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(tf_buffer)

        # broadcast static mount: wrist_3_link → camera_color_optical_frame
        static_bcaster = tf2_ros.StaticTransformBroadcaster()
        t = TransformStamped()
        t.header.stamp    = rospy.Time.now()
        t.header.frame_id = "wrist_3_link"
        t.child_frame_id  = intr.frame_id

        t.transform.translation.x = -0.032500
        t.transform.translation.y = -0.076287
        t.transform.translation.z =   0.032679 - 0.001930  # 30.749 mm to the actual focal point - 1,93mm from glass to the focal point

        q = quaternion_from_euler(
            math.radians(-8.0),  # roll
            0.0,                 # pitch
            0.0                  # yaw
        )
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        static_bcaster.sendTransform(t)
        rospy.loginfo("[XFORM] static wrist_3_link→%s published", intr.frame_id)

def raw_cb(msg: Float32MultiArray):
    global intr, marker_pub, tf_buffer, cam_model
    if intr is None or tf_buffer is None:
        rospy.logwarn_throttle(5.0, "[XFORM] waiting for camera_info…")
        return

    arr = np.array(msg.data, dtype=float).reshape(-1,4)
    best = None
    for u, v, r_px, D_mm in arr:
        Z_cam = D_mm * 0.001
        if Z_cam <= 0.01:
            continue
        r_m = (r_px * Z_cam) / intr.fx
        Zc = Z_cam + r_m
        if best is None or Zc < best[0]:
            best = (Zc, int(u), int(v), r_m)

    if best is None:
        rospy.loginfo_throttle(5.0, "[XFORM] no valid tomato")
        return

    Zc, u, v, r_m = best

    # project into camera frame
    ray = cam_model.projectPixelTo3dRay((u, v))
    X = ray[0] * Zc
    Y = ray[1] * Zc
    Z = ray[2] * Zc

    # pose in optical frame
    ps = PoseStamped()
    ps.header.stamp    = rospy.Time.now()
    ps.header.frame_id = intr.frame_id
    ps.pose.position.x = X
    ps.pose.position.y = Y
    ps.pose.position.z = Z
    ps.pose.orientation.w = 1.0

    # transform to wrist_3_link
    try:
        ps_wrist = tf_buffer.transform(ps, "wrist_3_link", rospy.Duration(0.5))
        rospy.loginfo("Tomato in wrist_3_link: x=%.4f y=%.4f z=%.4f",
                      ps_wrist.pose.position.x,
                      ps_wrist.pose.position.y,
                      ps_wrist.pose.position.z)
    except Exception as e:
        rospy.logwarn(f"[XFORM] TF→wrist_3_link failed: {e}")

    # then to base_link for RViz marker
    try:
        ps_base = tf_buffer.transform(ps, "base_link", rospy.Duration(0.5))
    except Exception as e:
        rospy.logwarn(f"[XFORM] TF→base_link failed: {e}")
        return

    m = Marker()
    m.header.frame_id = "base_link"
    m.header.stamp    = ps_base.header.stamp
    m.ns              = "tomato"
    m.id              = 0
    m.type            = Marker.SPHERE
    m.action          = Marker.ADD
    m.pose            = ps_base.pose
    diameter = 2 * r_m
    m.scale.x = m.scale.y = m.scale.z = diameter
    m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.0, 0.0, 0.8
    m.lifetime = rospy.Duration(0)
    marker_pub.publish(m)

def main():
    global marker_pub
    rospy.init_node("coordinate_transform", anonymous=False)

    rospy.Subscriber("/camera/aligned_depth_to_color/camera_info",
                     CameraInfo, camera_info_cb, queue_size=1)
    rospy.Subscriber("/yolo_raw_detections", Float32MultiArray,
                     raw_cb, queue_size=1)
    marker_pub = rospy.Publisher("/tomato_position", Marker,
                                 queue_size=1)

    rospy.loginfo("[XFORM] node started")
    rospy.spin()

if __name__ == "__main__":
    main()
